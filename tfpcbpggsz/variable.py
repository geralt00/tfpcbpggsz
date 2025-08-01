"""
This module implements classes and methods to manage the variables in fitting.
"""

import abc
import contextlib
import random
import warnings

import numpy as np
import sympy as sy

from tfpcbpggsz.generator.config import get_config, regist_config

from tfpcbpggsz.params_trans import ParamsTrans
from tfpcbpggsz.tensorflow_wrapper import tf


class AbsParameterGenerator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self):
        raise NotImplemented

    def force_in_range(self, a, b):
        idx = 0
        while True:
            idx += 1
            if idx > 10000:
                warnings.warn("max generate step, force to range center")
                x = (a + b) / 2
                break
            x = self()
            if a is not None and x < a:
                continue
            if b is not None and x > b:
                continue
            return x
        return x


class FunParameterGenerator(AbsParameterGenerator):
    def __init__(self, fun, *args, **kwargs):
        self.fun = fun
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.fun(*self.args, **self.kwargs)


class FixVar(AbsParameterGenerator):
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


def _gen_from_random(name, *args, module=tf.random, **kwargs):
    return FunParameterGenerator(
        getattr(module, name), *args, shape=(), **kwargs
    )


def combineVM(vm1, vm2, name="", same_list=None):
    """
    This function combines two VarsManager objects into one. (WIP)

    :param name: The name of this combined VarsManager
    :param same_list: To make some variables in the two VarsManager to be the same. E.g. if ``same_list = ["var",["var1","var2"]]``, then "var" in vm1 and vm2 will be the same, and "var1" in vm1 and "var2" in vm2 will be the same.
    """
    if same_list is None:
        same_list = []
    if vm1.name == vm2.name:
        raise Exception(
            "The two VarsManager to be combined have the same name."
        )
    vm = VarsManager(name, vm1.dtype)

    for i in vm1.variables:
        ii = vm1.name + i
        vm.variables[ii] = vm1.variables[i]  # vm.variables
        if vm.variables[ii].trainable:
            vm.trainable_vars.append(ii)  # vm.trainable_vars
        if i in vm1.bnd_dic:
            vm.bnd_dic[ii] = vm1.bnd_dic[i]  # vm.bnd_dic
    for i in vm2.variables:
        ii = vm2.name + i
        vm.variables[ii] = vm2.variables[i]
        if vm.variables[ii].trainable:
            vm.trainable_vars.append(ii)
        if i in vm2.bnd_dic:
            vm.bnd_dic[ii] = vm2.bnd_dic[i]

    for i in vm1.complex_vars:
        ii = vm1.name + i
        vm.complex_vars[ii] = vm1.complex_vars[i]  # vm.complex_vars
    for i in vm2.complex_vars:
        ii = vm2.name + i
        vm.complex_vars[ii] = vm2.complex_vars[i]

    for i in vm1.same_list:
        vm.same_list.append([vm1.name + j for j in i])  # vm.same_list
    for i in vm2.same_list:
        vm.same_list.append([vm2.name + j for j in i])

    for V in vm1.var_head:
        vm.var_head[V] = [vm1.name + i for i in vm1.var_head[V]]  # vm.var_head
        V.name = vm1.name + V.name
        V.vm = vm
        V.all_name_list = V.init_name_list()
    for V in vm2.var_head:
        vm.var_head[V] = [vm2.name + i for i in vm2.var_head[V]]
        V.name = vm2.name + V.name
        V.vm = vm
        V.all_name_list = V.init_name_list()

    for i in same_list:
        if type(i) == str:
            vm.set_same([vm1.name + i, vm2.name + i])
        else:
            vm.set_same([vm1.name + i[0], vm2.name + i[1]])

    del vm1, vm2
    return vm


regist_config("polar", True)
regist_config("multigpu", False)


class VarsManager(object):
    """
    This class provides methods to operate the variables in fitting. Every variable is a 1-d **tf.Variable** of
    **dtype** (**tf.float64** by default).

    All variables are stored in a dictionary **self.variables**. The indices of the dictionary are the variables' names,
    so name property in **tf.Variable** does not matter. All methods intended to change the variables are operating
    **self.variables** directly.

    Besides, all trainable variables' names will be stored in a list **self.trainable_vars**.
    """

    def __init__(self, name="", dtype=tf.float64, multi_gpu=None):
        self.name = name
        self.dtype = dtype
        self.polar = get_config("polar")
        self.multigpu = (
            get_config("multigpu") if multi_gpu is None else multi_gpu
        )
        if self.multigpu:
            self.strategy = tf.distribute.MirroredStrategy()
        else:
            self.strategy = None
        self.variables = {}  # {name:tf.Variable,...}
        self.trainable_vars = []  # [name,...]
        self.complex_vars = {}  # {name:polar(bool),...}
        self.same_list = []  # [[name1,name2],...]
        self.same_var_map = {}
        self.mask_vars = {}
        self.pre_trans = {}

        self.bnd_dic = {}  # {name:(a,b),...}

        self.var_head = (
            {}
        )  # {head:[name1,name2],...} It's operated directly by Variable objects

        self.init_val = {}
        self.random_generator = {}

    def add_real_var(self, name, value=None, range_=None, trainable=True):
        """
        Add a real variable named **name** into **self.variables**. If **value** and **range_** are not provided, the initial value
        is set to be a uniform random number between 0 and 1.

        :param name: The name of the variable, the index of this variable in **self.variables**
        :param value: The initial value.
        :param range_: Length-2 array. It's useless if **value** is given. Otherwise the initial value is set to be a uniform random number between **range_[0]** and **range_[0]**.
        :param trainable: Boolean. If it's **True**, the variable is trainable while fitting.
        """
        if self.multigpu:
            with self.strategy.scope():
                self._add_real_var(name, value, range_, trainable)
        else:
            self._add_real_var(name, value, range_, trainable)

    def _add_real_var(self, name, value=None, range_=None, trainable=True):
        if name in self.variables:  # not a new var
            return
            # if name in self.trainable_vars:
            #    self.trainable_vars.remove(name)
            # warnings.warn("Overwrite variable {}!".format(name))
        if value is None:
            if range_ is None:  # random [0,1]
                range_ = [0.0, 1.0]
            self.random_generator[name] = _gen_from_random(
                "uniform", minval=range_[0], maxval=range_[1], dtype=self.dtype
            )
            self.variables[name] = tf.Variable(
                self.random_generator[name](), trainable=trainable
            )
        else:  # constant value
            # if name in self.bnd_dic:
            # value = self.bnd_dic[name].get_y2x(value)
            if hasattr(value, "__len__"):
                mu = value[0]
                sigma = value[1]
                self.random_generator[name] = _gen_from_random(
                    "normal", mean=mu, stddev=sigma, dtype=self.dtype
                )
            else:
                self.random_generator[name] = FixVar(value)
            self.variables[name] = tf.Variable(
                self.random_generator[name](),
                dtype=self.dtype,
                trainable=trainable,
            )
            self.init_val[name] = value

        if trainable:
            self.trainable_vars.append(name)

    def add_complex_var(
        self, name, polar=None, trainable=True, fix_vals=(1.0, 0.0)
    ):
        """
        Add a complex variable. Two real variables named **name+'r'** and **name+'i'** will be added into
        **self.variables**. The initial values will be given automatically according to its form of coordinate.

        :param name: The name of the complex variable.
        :param polar: Boolean. If it's **True**, **name+'r'** and **name+'i'** are defined in polar coordinate; otherwise they are defined in Cartesian coordinate.
        :param trainable: Boolean. If it's **True**, real variables **name+'r'** and **name+'i'** will be trainable.
        :param fix_vals: Length-2 array. If **trainable=False**, the fixed values for **name+'r'** and **name+'i'** are **fix_vals[0]**, **fix_vals[1]** respectively.
        """
        if polar is None:
            polar = self.polar
        var_r = name + "r"
        var_i = name + "i"
        if trainable:
            if polar:
                self.add_real_var(name=var_r, range_=(0.0, 2.0))
                self.add_real_var(name=var_i, range_=(-np.pi, np.pi))
            else:
                self.add_real_var(name=var_r, range_=(-1, 1))
                self.add_real_var(name=var_i, range_=(-1, 1))
        else:
            self.add_real_var(name=var_r, value=fix_vals[0], trainable=False)
            self.add_real_var(name=var_i, value=fix_vals[1], trainable=False)
        self.complex_vars[name] = polar

    def add_cartesiancp_var(
        self, name, polar=None, trainable=True, fix_vals=(1.0, 0.0, 0.0, 0.0)
    ):
        """
        Add a complex variable. Two real variables named **name+'r'** and **name+'i'** will be added into
        **self.variables**. The initial values will be given automatically according to its form of coordinate.

        :param name: The name of the complex variable.
        :param polar: Boolean. If it's **True**, **name+'r'** and **name+'i'** are defined in polar coordinate; otherwise they are defined in Cartesian coordinate.
        :param trainable: Boolean. If it's **True**, real variables **name+'r'** and **name+'i'** will be trainable.
        :param fix_vals: Length-4 array. If **trainable=False**, the fixed values for **name+'r'** and **name+'i'** are **fix_vals[0]**, **fix_vals[1]** respectively.
        """
        if polar is None:
            polar = self.polar
        # if chargeconjugate is None:
        # chargeconjugate = self.chargeconjugate
        var_r = name + "r"
        var_i = name + "i"
        var_deltar = name + "deltar"
        var_deltai = name + "deltai"
        if trainable:
            self.add_real_var(name=var_r, range_=(-1, 1))
            self.add_real_var(name=var_i, range_=(-1, 1))
            self.add_real_var(name=var_deltar, range_=(-1, 1))
            self.add_real_var(name=var_deltai, range_=(-1, 1))
        else:
            self.add_real_var(name=var_r, value=fix_vals[0], trainable=False)
            self.add_real_var(name=var_i, value=fix_vals[1], trainable=False)
            self.add_real_var(
                name=var_deltar, value=fix_vals[2], trainable=False
            )
            self.add_real_var(
                name=var_deltai, value=fix_vals[3], trainable=False
            )

        self.complex_vars[name] = polar

    def remove_var(self, name):
        """
        Remove a variable from **self.variables**. More specifically, two variables (**name+'r'** and **name+'i'**)
        will be removed if it's complex.

        :param name: The name of the variable
        """
        if name in self.complex_vars:
            del self.complex_vars[name]
            self.remove_var(name + "r")
            self.remove_var(name + "i")
        else:
            if self.variables[name].trainable:
                if name in self.trainable_vars:
                    self.trainable_vars.remove(name)
            for l in self.same_list:
                if name in l:
                    l.remove(name)
            if name in self.bnd_dic:
                del self.bnd_dic[name]
            del self.variables[name]

    def rename_var(self, name, new_name, cplx=False):
        """
        Rename a variable.

        :param name: Name of the variable
        :param new_name: New name
        :param cplx: Boolean. Users should indicate if this variable is complex or not.
        """
        if cplx:
            self.complex_vars[new_name] = self.complex_vars[name]
            del self.complex_vars[name]
            name_r = name + "r"
            name_i = name + "i"
            new_name_r = new_name + "r"
            new_name_i = new_name + "i"
            self.rename_var(name_r, new_name_r)
            self.rename_var(name_i, new_name_i)
        else:
            if self.variables[name].trainable:
                self.trainable_vars.remove(name)
                self.trainable_vars.append(new_name)
            for l in self.same_list:
                if name in l:
                    l.remove(name)
                    l.append(new_name)
            if name in self.bnd_dic:
                self.bnd_dic[new_name] = self.bnd_dic[name]
                del self.bnd_dic[name]
            self.variables[new_name] = self.variables[name]
            del self.variables[name]

    def regenerate_var(self, name, force=False):
        if force or name in self.trainable_vars:
            self.variables[name].assign(self.random_generator[name]())

    def refresh_vars(self, init_val=None, bound_dic=None):
        """
        Refresh all trainable variables
        """
        if bound_dic is not None:
            self.set_bound(bound_dic)
        for name in self.trainable_vars:
            if init_val is not None and name in init_val:
                val = init_val[name]
            elif bound_dic is not None and name in bound_dic:
                val = self.random_generator[name].force_in_range(
                    *bound_dic[name]
                )
            else:
                val = self.random_generator[name]()
            self.variables[name].assign(val)

    def smear_vars(self, error_matrix):
        mean = self.get_all_val()
        new_mean = np.random.multivariate_normal(mean, error_matrix)
        return dict(zip(self.trainable_vars, new_mean))

    def set_fix(self, name, value=None, unfix=False):
        """
        Fix or unfix a variable (change the trainability)
        :param name: The name of the variable
        :param value: The fixed value. It's useless if **unfix=True**.
        :param unfix: Boolean. If it's **True**, the variable will become trainable rather than be fixed.
        """
        if name in self.same_var_map:
            name = self.same_var_map[name]
        if value is None:
            if name in self.variables:
                value = self.variables[name].value
            else:
                value = 0.0
            if callable(value):
                value = value()
        else:
            if name in self.bnd_dic:
                value = self.bnd_dic[name].get_y2x(value)
        if name in self.variables:
            self.variables[name].assign(value)
            self.variables[name]._trainable = unfix
        else:
            warnings.warn("no veriable named as {} in free".format(name))
            return
        if unfix:
            if name in self.trainable_vars:
                warnings.warn("{} has been freed already!".format(name))
            else:
                self.trainable_vars.append(name)
        else:
            if name in self.trainable_vars:
                self.trainable_vars.remove(name)
            else:
                warnings.warn("{} has been fixed already!".format(name))

    def set_bound(self, bound_dic, func=None, overwrite=False):
        """
        Set boundary for the trainable variables. The variables will be constrained in their ranges while fitting.

        :param bound_dic: Dictionary. E.g. **{"name1":(-1.0,1.0), "name2":(None,1.0)}**. In this example, **None** means it has no lower limit.
        :param func: String. Users can provide a string to describe the transforming function. For details, refer to class **tfpcbpggsz.variable.Bound**.
        :param overwrite: Boolean. If it's ``True``, the program will not throw a warning when overwrite a variable with the same name.
        """
        for name in bound_dic:
            if name in self.bnd_dic:
                if not overwrite:
                    warnings.warn("Overwrite bound of {}!".format(name))
            self.bnd_dic[name] = Bound(*bound_dic[name], func=func)
            a, b = bound_dic[name]
            if a is None and b is not None:
                self.random_generator[name] = FunParameterGenerator(
                    lambda: b - np.random.chisquare(df=1)
                )
            elif a is not None and b is None:
                self.random_generator[name] = FunParameterGenerator(
                    lambda: a + np.random.chisquare(df=1)
                )
            elif a is not None and b is not None:
                self.random_generator[name] = _gen_from_random(
                    "uniform", minval=a, maxval=b, dtype=self.dtype
                )
            else:  # not overwrite
                pass
            if name in self.variables:
                has_same = False
                for i in self.same_list:
                    if name in i[1:]:
                        has_same = True
                        break
                if has_same:
                    continue
                # val = self.get(name).numpy()
                # self.set(name, self.bnd_dic[name].get_y2x(val))

    def _remove_bound(self, name):
        if name in self.variables:
            has_same = False
            for i in self.same_list:
                if name in i[1:]:
                    has_same = True
                    break
            if not has_same:
                value = self.get(name, val_in_fit=False)
                # self.set(name, value)
        del self.bnd_dic[name]

    def remove_bound(self):
        """
        Remove a boundary for a variable
        """
        bnd_dic = self.bnd_dic.copy()
        for i in bnd_dic:
            self._remove_bound(i)
        return bnd_dic

    def set_share_r(self, name_list):  # name_list==[name1,name2,...]
        """
        If some complex variables want to share their radia variable while their phase variable are still different.
        Users can set this type of constrain using this method.

        :param name_list: List of strings. Note the strings should be the name of the complex variables rather than of their radium parts.
        """
        self.xy2rp_all(name_list)
        name_r_list = [name + "r" for name in name_list]
        self.set_same(name_r_list)
        for name in name_r_list:
            self.complex_vars[name[:-1]] = True  # name_r_list

    def set_same(self, name_list, cplx=False):
        """
        Set some variables to be the same.

        :param name_list: List of strings. Name of the variables.
        :param cplx: Boolean. Whether the variables are complex or real.
        """
        if cplx:
            self.set_same([i + "r" for i in name_list])
            self.set_same([i + "i" for i in name_list])
            return
        tmp_list = []
        head_list = []
        for name in name_list:
            for add_list in self.same_list:
                if name not in self.variables:
                    continue
                if name in add_list:
                    tmp_list += add_list
                    head_list += [add_list[0]]
                    self.same_list.remove(add_list)
                    break

        # use head to avoid duplicate setting
        new_name_list = head_list
        for i in name_list:
            if i not in tmp_list:
                new_name_list.append(i)

        # remove duplicate items
        for i in tmp_list:
            if i not in name_list:
                name_list.append(i)

        def same_real(name_list):
            name_list = [i for i in name_list if i in self.variables]
            if len(name_list) == 0:
                return
            var = self.variables[name_list[0]]
            for name in name_list[1:]:
                if name in self.trainable_vars:
                    self.trainable_vars.remove(name)
                else:
                    # if one is untrainable, the others will all be untrainable
                    var2 = self.variables.get(name, None)
                    if var2 is not None:
                        if name_list[0] in self.trainable_vars:
                            self.trainable_vars.remove(name_list[0])
            for name in name_list:
                self.variables[name] = var

        if cplx:
            same_real([name + "r" for name in new_name_list])
            same_real([name + "i" for name in new_name_list])
        else:
            same_real(new_name_list)
        self.same_list.append(name_list)
        for i in name_list:
            self.same_var_map[i] = name_list[0]

    def get(self, name, val_in_fit=True):
        """
        Get a real variable. If ``val_in_fit is True``, this is the variable used in fitting, not considering its boundary transformation.

        :param name: String
        :return: tf.Variable
        """
        if name not in self.variables:
            raise Exception("{} not found".format(name))
        if not val_in_fit or name not in self.bnd_dic:
            value = self.variables[name]
            if name in self.pre_trans:
                value = self.pre_trans[name](self.variables)
            return value.numpy()  # tf.Variable
        else:
            return self.bnd_dic[name].get_y2x(self.variables[name].numpy())

    def read(self, name):
        val = self.variables[name]
        if name in self.mask_vars:
            val = tf.stop_gradient(tf.cast(self.mask_vars[name], val.dtype))
        if name in self.pre_trans:
            trans = self.pre_trans[name]
            val = trans(self.variables)
        return val

    def set(self, name, value, val_in_fit=True):
        """
        Set value for a real variable. If ``val_in_fit is True``, this is the variable used in fitting, not considering its boundary transformation.

        :param name: String
        :param value: Real number
        """
        if val_in_fit and name in self.bnd_dic:
            value = self.bnd_dic[name].get_x2y(value)
        if name in self.variables:
            if name in self.pre_trans:
                trans = self.pre_trans[name]
                value = trans.inverse(value)
            if value is not None:
                self.variables[name].assign(value)
        else:
            warnings.warn("{} not found".format(name))

    def rp2xy(self, name):
        """
        Transform a complex variable into Cartesian coordinate.
        :param name: String
        """
        if self.complex_vars[name] != True:  # if not polar (already xy)
            return
        r = self.variables[name + "r"]
        p = self.variables[name + "i"]
        x = r * tf.cos(p)
        y = r * tf.sin(p)
        self.variables[name + "r"].assign(x)
        self.variables[name + "i"].assign(y)
        self.complex_vars[name] = False
        for l in self.same_list:
            if name in l:
                for i in l:
                    self.complex_vars[i] = False
                break

    def xy2rp(self, name):
        """
        Transform a complex variable into polar coordinate.
        :param name: String
        """
        if self.complex_vars[name] != False:  # if already polar
            return
        x = self.variables[name + "r"]
        y = self.variables[name + "i"]
        r = tf.sqrt(x * x + y * y)
        p = tf.atan2(y, x)
        self.variables[name + "r"].assign(r)
        self.variables[name + "i"].assign(p)
        self.complex_vars[name] = True
        for l in self.same_list:
            if name in l:
                for i in l:
                    self.complex_vars[i] = True
                break

    @property
    def trainable_variables(self):
        """
        List of tf.Variable. It is similar to **tf.keras.Model.trainable_variables**.
        """
        vars_list = []
        for name in self.trainable_vars:
            vars_list.append(self.variables[self.same_var_map.get(name, name)])
        return vars_list

    def get_all_val(self, val_in_fit=False):  # if bound transf var
        """
        Get the values of all trainable variables.

        :param val_in_fit: Boolean. If it's **True**, the values will be the ones that are actually used in fitting (thus may not be the physical values because of the boundary transformation).
        :return: List of real numbers.
        """
        vals = []
        for name in self.trainable_vars:
            xval = self.get(name, val_in_fit)
            vals.append(xval)
        return vals  # list (for list of tf.Variable use self.trainable_variables; for dict of all vars, use self.variables)

    def get_all_dic(self, trainable_only=False):
        """
        Get a dictionary of all variables.

        :param trainable_only: Boolean. If it's **True**, the dictionary only contains the trainable variables.
        :return: Dictionary
        """
        # self.std_polar_all()
        dic = {}
        if trainable_only:
            for i in self.trainable_vars:
                val = self.read(i).numpy()
                # if i in self.bnd_dic:
                #     val = self.bnd_dic[i].get_y2x(val)
                dic[i] = val
        else:
            for i in self.variables:
                val = self.read(i).numpy()
                # if i in self.bnd_dic:
                #    val = self.bnd_dic[i].get_y2x(val)
                dic[i] = val
        return dic

    def set_all(self, vals, val_in_fit=False):  # use either dict or list
        """
        Set values for some variables.

        :param vals: It can be either a dictionary or a list of real numbers. If it's a list, the values correspond to all trainable variables in order.
        """
        if type(vals) == dict:
            for name in vals:
                self.set(name, vals[name], val_in_fit=val_in_fit)
        else:
            i = 0
            for name in self.trainable_vars:
                self.set(name, vals[i], val_in_fit)
                i += 1

    def rp2xy_all(self, name_list=None):
        """
        If **name_list** is not provided, this method will transform all complex variables into Cartesian coordinate.

        :param name_list: List of names of complex variables
        """
        if not name_list:
            name_list = self.complex_vars
        for name in name_list:
            self.rp2xy(name)
        self.polar = False

    def xy2rp_all(self, name_list=None):
        """
        If **name_list** is not provided, this method will transform all complex variables into polar coordinate.

        :param name_list: List of names of complex variables
        """
        if not name_list:
            name_list = self.complex_vars
        for name in name_list:
            self.xy2rp(name)
        self.polar = True

    @staticmethod
    def _std_polar_angle(p, a=-np.pi, b=np.pi):
        return (p - a) % (b - a) + a

    def std_polar(self, name):
        """
        Transform a complex variable into standard polar coordinate, which mean its radium part is positive, and its
        phase part is between :math:`-\\pi` to :math:`\\pi`.
        :param name: String
        """
        self.xy2rp(name)
        r = self.variables[name + "r"]
        p = self.variables[name + "i"]
        if r < 0:
            r.assign(tf.abs(r))
            if type(self.complex_vars[name]) == list:
                for name_r in self.complex_vars[name]:
                    self.variables[name_r[:-1] + "i"].assign_add(np.pi)
            else:
                p.assign_add(np.pi)
        self._std_polar_angle(p)

    def std_polar_all(self):  # std polar expression: r>0, -pi<p<pi
        """
        Transform all complex variables into standard polar coordinate.
        """
        for name in self.complex_vars:
            self.std_polar(name)

    def standard_complex(self):
        for k, v in self.complex_vars.items():
            ## TODO complex with constrains
            if isinstance(v, list):
                continue
            if not v:
                continue
            has_constrains = False
            for i in self.same_list:
                if k + "r" in i or k + "i" in i:
                    has_constrains = True
            if k + "r" in self.bnd_dic:
                has_constrains = True
            if k + "i" in self.bnd_dic:
                has_constrains = True
            if has_constrains:
                continue
            self.std_polar(k)

    def trans_params(self, polar):
        """
        Transform all complex variables into either polar coordinate or Cartesian coordinate.

        :param polar: Boolean
        """
        if polar:
            self.std_polar_all()
        else:
            self.rp2xy_all()

    def set_trans_var(self, xvals):
        """
        :math:`y = y(x)`

        :param fcn_grad: The return of class **tfpcbpggsz.model**???
        :return:
        """

        xvals = np.array(xvals)
        yvals = xvals.copy()
        for i, name in enumerate(self.trainable_vars):
            if name in self.bnd_dic:
                yvals[i] = self.bnd_dic[name].get_x2y(xvals[i])
        self.set_all(yvals)

    def trans_fcn_grad(self, fcn_grad):  # bound transform fcn and grad
        """
        :math:`F(x)=F(y(x))`, :math:`G(x)=\\frac{dF}{dx}=\\frac{dF}{dy}\\frac{dy}{dx}`

        :param fcn_grad: The return of class **tfpcbpggsz.model**???
        :return:
        """

        def fcn_t(xvals):
            xvals = np.array(xvals)
            yvals = xvals.copy()
            dydxs = []
            i = 0
            for name in self.trainable_vars:
                if name in self.bnd_dic:
                    yvals[i] = self.bnd_dic[name].get_x2y(xvals[i])
                    dydxs.append(self.bnd_dic[name].get_dydx(xvals[i]))
                else:
                    dydxs.append(1)
                i += 1
            fcn, grad_yv = fcn_grad(yvals)
            grad = np.array(grad_yv) * dydxs
            return fcn, grad

        return fcn_t

    def trans_grad_hessp(self, f):  # bound transform fcn and grad
        """
        :math:`F(x)=F(y(x))`, :math:`G(x)=\\frac{dF}{dx}=\\frac{dF}{dy}\\frac{dy}{dx}`

        :param fcn_grad: The return of class **tfpcbpggsz.model**???
        :return:
        """

        def f_wrap(xvals, p):
            xvals = np.array(xvals)
            yvals = xvals.copy()
            dydxs = []
            dydxs2 = []
            i = 0
            for name in self.trainable_vars:
                if name in self.bnd_dic:
                    yvals[i] = self.bnd_dic[name].get_x2y(xvals[i])
                    dydxs.append(self.bnd_dic[name].get_dydx(xvals[i]))
                    dydxs2.append(self.bnd_dic[name].get_d2ydx2(xvals[i]))
                else:
                    dydxs.append(1)
                    dydxs2.append(0)
                i += 1

            dydxs = np.array(dydxs)
            dydxs2 = np.array(dydxs2)

            # print(xvals.shape, p.shape, dydxs.shape, len(self.trainable_vars))

            grad_yv, hessp_yv = f(yvals, p * dydxs)
            grad = np.array(grad_yv) * dydxs
            # print("trans", p, grad, hessp_yv * dydxs + grad_yv * dydxs2)
            return grad, hessp_yv * dydxs + grad_yv * dydxs2 * p

        return f_wrap

    def trans_f_grad_hess(self, f):  # bound transform fcn and grad
        """
        :math:`F(x)=F(y(x))`, :math:`G(x)=\\frac{dF}{dx}=\\frac{dF}{dy}\\frac{dy}{dx}`

        :param fcn_grad: The return of class **tfpcbpggsz.model**???
        :return:
        """

        def f_wrap(xvals):
            xvals = np.array(xvals)
            yvals = xvals.copy()
            dydxs = []
            dydxs2 = []
            i = 0
            for name in self.trainable_vars:
                if name in self.bnd_dic:
                    yvals[i] = self.bnd_dic[name].get_x2y(xvals[i])
                    dydxs.append(self.bnd_dic[name].get_dydx(xvals[i]))
                    dydxs2.append(self.bnd_dic[name].get_d2ydx2(xvals[i]))
                else:
                    dydxs.append(1)
                    dydxs2.append(0)
                i += 1

            dydxs = np.array(dydxs)
            dydxs2 = np.array(dydxs2)

            # print(yvals) # , xvals.shape, p.shape, dydxs.shape, len(self.trainable_vars))

            f2, grad_yv, hessp_yv = f(yvals)
            grad = np.array(grad_yv) * dydxs
            # print("trans", p, grad, hessp_yv * dydxs + grad_yv * dydxs2)
            return (
                f2,
                grad,
                dydxs[:, None] * hessp_yv * dydxs[None, :]
                + np.diag(grad_yv * dydxs2),
            )

        return f_wrap

    def trans_error_matrix(self, hess_inv, xvals):
        """
        Bound trans for error matrix
        :math:`F(x)=F(y(x))`, :math:`V_y = y' V_x y'`

        :return:
        """
        xvals = np.array(xvals)
        dydxs = []
        for i, name in enumerate(self.trainable_vars):
            if name in self.bnd_dic:
                dydxs.append(self.bnd_dic[name].get_dydx(xvals[i]))
            else:
                dydxs.append(1)
        dydx = np.array(dydxs)
        hess_inv = dydx[:, None] * np.array(hess_inv) * dydx[None, :]
        return hess_inv

    def batch_sum_var(self, fun, data, batch=65000):
        from tfpcbpggsz.data import batch_sum

        var = self.trainable_variables
        return batch_sum(
            lambda x: SumVar.from_call(fun, var, x), data, batch=batch
        )

    @contextlib.contextmanager
    def error_trans(self, err_matrix):
        with ParamsTrans(self, err_matrix).trans() as f:
            yield f

    @contextlib.contextmanager
    def temp_params(self, params):
        old_params = {i: self.get(i) for i in params.keys()}
        self.set_all(params)
        yield
        self.set_all(old_params)

    @contextlib.contextmanager
    def mask_params(self, params):
        old_mask = self.mask_vars
        self.mask_vars = params
        yield
        self.mask_vars = old_mask

    def minimize(self, fcn, jac=True, method="BFGS", mini_kwargs={}):
        """
        minimize a give function
        """
        if hasattr(fcn, "nll_grad"):
            f = fcn.nll_grad
        else:

            def f(x):
                self.set_all(x)
                with tf.GradientTape() as tape:
                    y = fcn()
                g = tape.gradient(
                    y, self.trainable_variables, unconnected_gradients="zero"
                )
                return float(y), np.array([float(i) for i in g])

        x0 = self.get_all_val(True)

        f2 = self.trans_fcn_grad(f)
        if isinstance(method, str):
            from scipy.optimize import minimize as mini

            if jac != True:

                def f(x):
                    self.set_all(x)
                    y = fcn()
                    return float(y)

                ret = mini(
                    f, np.array(x0), jac=jac, method=method, **mini_kwargs
                )
            else:
                ret = mini(
                    f2, np.array(x0), jac=jac, method=method, **mini_kwargs
                )
        else:
            ret = method(f2, np.array(x0), **mini_kwargs)
        self.set_all(ret.x, val_in_fit=True)
        ret.x = np.array(self.get_all_val())
        if isinstance(ret.hess_inv, np.ndarray):
            ret.hess_inv = self.trans_error_matrix(ret.hess_inv, ret.x)
        else:
            ret.hess_inv = None
        return ret

    def minimize_error(self, fcn, fit_result):
        if hasattr(fit_result, "hess_inv") and isinstance(
            fit_result.hess_inv, np.ndarray
        ):
            hess_inv = fit_result.hess_inv
        else:

            def f(x):
                self.set_all(x)
                with tf.GradientTape(persistent=True) as tape0:
                    with tf.GradientTape() as tape:
                        y = fcn()
                    g = tape.gradient(
                        y,
                        self.trainable_variables,
                        unconnected_gradients="zero",
                    )
                hs = []
                for gi in g:
                    gi_grad = tape0.gradient(
                        gi,
                        self.trainable_variables,
                        unconnected_gradients="zero",
                    )
                    hs.append([float(i) for i in gi_grad])
                del tape0
                return float(y), np.array([float(i) for i in g]), np.array(hs)

            _, _, hess = f(fit_result.x)
            hess_inv = np.linalg.inv(hess)
            fit_result.hess_inv = self.trans_error_matrix(
                hess_inv, fit_result.x
            )
        x_error = np.sqrt(np.diag(fit_result.hess_inv))
        return x_error


class Bound(object):
    """
    This class provides methods to implement the boundary constraint for a variable.
    It has dependence on `SymPy <https://www.sympy.org/en/index.html>`_ .
    The boundary-transforming function can transform a variable *x* defined in the real domain to a variable *y* defined
    in a limited range *(a,b)*. *y* should be the physical parameter but *x* is the one used while fitting.

    :param a: Real number. The lower boundary
    :param b: Real number. The upper boundary
    :param func: String. The boundary-transforming function. By default, if neither **a** or **b** is **None**, **func** is **"(b-a)*(sin(x)+1)/2+a"**; else if only **a** is **None**, **func** is **"b+1-sqrt(x**2+1)"**; else if only **b** is **None**, **func** is **"a-1+sqrt(x**2+1)"**; else **func** is **"x"**.

    **a**, **b**, **func** can be refered by **self.lower**, **self.upper**, **self.func**.
    """

    def __init__(self, a=None, b=None, func=None):
        if a is not None and b is not None and a > b:
            raise Exception("Lower bound is larger than upper bound!")
        self.lower = a
        self.upper = b
        if func:
            self.func = func  # from R (x) to a limited range (y) #Note: y is gls but x is the var in fitting
        else:
            if a is None:
                if b is None:
                    self.func = "x"
                else:
                    self.func = "b+1-sqrt(x**2+1)"
            else:
                if b is None:
                    self.func = "a-1+sqrt(x**2+1)"
                else:
                    self.func = "(b-a)*(sin(x)+1)/2+a"
        self.f, self.df, self.df2, self.inv = self.get_func()

    def __repr__(self):
        return "[" + str(self.lower) + ", " + str(self.upper) + "]"

    def __iter__(self):
        return iter((self.lower, self.upper))

    def get_func(self):  # init func string into sympy f(x) or f(y)
        """
        Initialize the function string into **sympy** objects.

        :return: **sympy** objects **f**, **df**, **inv**, which are the function, its derivative and its inverse function.
        """
        x, a, b, y = sy.symbols("x a b y")
        f = sy.sympify(self.func)
        f = f.subs(
            {
                a: self.lower if self.lower is not None else -1e9,
                b: self.upper if self.upper is not None else 1e9,
            }
        )
        df = sy.diff(f, x)
        df2 = sy.diff(df, x)
        inv = sy.solve(f - y, x)
        if hasattr(inv, "__len__"):
            inv = inv[-1]
        return f, df, df2, inv

    def get_x2y(self, val):  # var->gls
        """
        To derive *y* from *x*

        :param val: Real number *x*
        :return: Real number *y*
        """
        x = sy.symbols("x")
        return float(self.f.evalf(subs={x: val}))

    def get_y2x(self, val):  # gls->var
        """
        To derive *x* from *y*. *y* will be set to *a* if *y<a*, and *y* will be set to *b* if *y>b*.

        :param val: Real number *y*
        :return: Real number *x*
        """
        y = sy.symbols("y")
        if self.lower is not None and val < self.lower:
            val = self.lower
        elif self.upper is not None and val > self.upper:
            val = self.upper
        x = self.inv.evalf(subs={y: val})
        return complex(x).real

    def get_dydx(self, val):  # gradient in fitting: dNLL/dx = dNLL/dy * dy/dx
        """
        To calculate the derivative :math:`\\frac{dy}{dx}`.

        :param val: Real number *x*
        :return: Real number :math:`\\frac{dy}{dx}`
        """
        x = sy.symbols("x")
        return float(self.df.evalf(subs={x: val}))

    def get_d2ydx2(
        self, val
    ):  # gradient in fitting: dNLL/dx = dNLL/dy * dy/dx
        """
        To calculate the derivative :math:`\\frac{dy}{dx}`.

        :param val: Real number *x*
        :return: Real number :math:`\\frac{dy}{dx}`
        """
        x = sy.symbols("x")
        return float(self.df2.evalf(subs={x: val}))


def _get_val_from_index(val, index):
    for i in index:
        val = val[i]
    return val


def _shape_func(f, shape, name="", idx=[], **kwargs):
    if not shape:
        f(name, idx, **kwargs)
    else:
        for i in range(shape[0]):
            _shape_func(f, shape[1:], name + "_" + str(i), idx + [i], **kwargs)


regist_config("vm", VarsManager(dtype="float64"))


# vm = VarsManager(dtype=tf.float64)
class Variable(object):
    """
    This class has interface to **VarsManager**. It is convenient for users to define a group of real variables,
    since it may be more perceptually intuitive to define them together.

    By calling the instance of this class, it returns the value of this variable. The type is tf.Tensor.

    :param name: The name of the variable group
    :param shape: The shape of the group. E.g. for a 4*3*2 matrix, **shape** is **[4,3,2]**. By default, **shape** is [] for a real variable.
    :param cplx: Boolean. Whether the variable (or the variables) are complex or not.
    :param vm: VarsManager. It is by default the one automatically defined in the global scope by the program.
    :param overwrite: Boolean. If it's ``True``, the program will not throw a warning when overwrite a variable with the same name.
    :param kwargs: Other arguments that may be used when calling **self.real_var()** or **self.cplx_var()**
    """

    def __init__(
        self,
        name,
        shape=None,
        cplx=False,
        vm=None,
        overwrite=True,
        is_cp=False,
        **kwargs,
    ):
        if shape is None:
            shape = []
        if vm is None:
            vm = get_config("vm")
        self.vm = vm
        self.name = name
        for i in self.vm.var_head:
            if i.name == self.name:
                if not overwrite:
                    ex = 1
                    while True:
                        ex_name = i.name + "_" + str(ex)
                        judg = True
                        for ii in self.vm.var_head:
                            if ii.name == ex_name:
                                ex += 1
                                judg = False
                                break
                        if judg:
                            break
                    self.name = ex_name
                else:
                    for j in self.vm.var_head[i]:
                        self.vm.remove_var(j)
                    del self.vm.var_head[i]
                break
        self.vm.var_head[self] = []
        if type(shape) == int:
            shape = [shape]
        self.shape = shape
        self.cplx = cplx
        self.cp_effect = is_cp
        if is_cp:
            # print("Variable init cplx_cpvar" )
            self.cplx_cpvar(**kwargs)
        elif cplx:
            self.cplx_var(**kwargs)
        else:
            self.real_var(**kwargs)
        self.bound = None
        self.all_name_list = self.init_name_list()

    def real_var(self, value=None, range_=None, fix=False):
        """
        It implements interface to ``VarsManager.add_real_var()``, but supports variables that are not of non-shape.

        :param value: Real number. The value of all real components.
        :param range_: Length-2 array. The length of all real components.
        :param fix: Boolean. Whether the variable is fixed.
        """
        trainable = not fix

        def func(name, idx, **kwargs):
            self.vm.add_real_var(name, value, range_, trainable)
            self.vm.var_head[self].append(name)

        _shape_func(
            func,
            self.shape,
            self.name,
            value=value,
            range_=range_,
            trainable=trainable,
        )

    def cplx_var(self, polar=None, fix=False, fix_vals=(1.0, 0.0)):
        """
        It implements interface to ``VarsManager.add_complex_var()``, but supports variables that are not of non-shape.

        :param polar: Boolean. Whether the variable is defined in polar coordinate or in Cartesian coordinate.
        :param fix: Boolean. Whether the variable is fixed. It's enabled only if ``self.shape is None``.
        :param fix_vals: Length-2 tuple. The value of the fixed complex variable is ``fix_vals[0]+fix_vals[1]j``.
        """
        if not hasattr(fix_vals, "__len__"):
            fix_vals = [fix_vals, 0.0]

        def func(name, idx, **kwargs):
            trainable = not fix
            self.vm.add_complex_var(name, polar, trainable, fix_vals)
            self.vm.var_head[self].append(name)

        _shape_func(
            func, self.shape, self.name, polar=polar, fix_vals=fix_vals
        )

    def cplx_cpvar(
        self, polar=True, fix=False, fix_vals=(1.0, 0.0, 0.0, 0.0), value=0.0
    ):
        """
        It implements interface to ``VarsManager.add_complex_var()``, but supports variables that are not of non-shape.

        :param polar: Boolean. Whether the variable is defined in polar coordinate or in Cartesian coordinate.
        :param fix: Boolean. Whether the variable is fixed. It's enabled only if ``self.shape is None``.
        :param fix_vals: Length-4 tuple. The value of the fixed complex variable is ``fix_vals[0]+fix_vals[1]j``.
        """
        if not hasattr(fix_vals, "__len__"):
            fix_vals = [fix_vals, 0.0, 0.0, 0.0]

        def func(name, idx, **kwargs):
            trainable = not fix
            # print("Variable cplx_cpvar")
            self.vm.add_cartesiancp_var(name, polar, trainable, fix_vals)
            self.vm.var_head[self].append(name)

        _shape_func(
            func,
            self.shape,
            self.name,
            polar=polar,
            fix_vals=fix_vals,
            value=value,
        )

    def __repr__(self):
        return self.name

    @property
    def value(self):
        """
        :return: Ndarray of ``self.shape``.
        """
        return tf.Variable(self()).numpy()

    def set_value(self, value, index=None):
        if index is not None:
            assert len(index) == len(self.shape)
            var_name = self.name
            for i in index:
                var_name += "_" + str(index[i])
            if self.cp_effect == True:
                self.vm.set(var_name + "r", value[0])
                self.vm.set(var_name + "i", value[1])
                self.vm.set(var_name + "deltar", value[2])
                self.vm.set(var_name + "deltai", value[3])
            elif self.cplx == True:
                self.vm.set(var_name + "r", value[0])
                self.vm.set(var_name + "i", value[1])
            else:
                self.vm.set(var_name, value)
        else:
            value = np.array(value)

            def _get_val_from_index(val, index):
                for i in index:
                    val = val[i]
                return val

            if self.cp_effect == True:
                if value.shape[:-1] == ():

                    def func(name, idx, **kwargs):
                        self.vm.set(name + "r", value[0])
                        self.vm.set(name + "i", value[1])
                        self.vm.set(name + "deltar", value[2])
                        self.vm.set(name + "deltai", value[3])

                elif value.shape[:-1] == tuple(self.shape):

                    def func(name, idx, **kwargs):
                        self.vm.set(
                            name + "r", _get_val_from_index(value, idx)[0]
                        )
                        self.vm.set(
                            name + "i", _get_val_from_index(value, idx)[1]
                        )
                        self.vm.set(
                            name + "deltar", _get_val_from_index(value, idx)[2]
                        )
                        self.vm.set(
                            name + "deltai", _get_val_from_index(value, idx)[3]
                        )

                else:
                    raise Exception(
                        "The shape of value should be ", self.shape
                    )
            elif self.cplx == True:
                if value.shape[:-1] == ():

                    def func(name, idx, **kwargs):
                        self.vm.set(name + "r", value[0])
                        self.vm.set(name + "i", value[1])

                elif value.shape[:-1] == tuple(self.shape):

                    def func(name, idx, **kwargs):
                        self.vm.set(
                            name + "r", _get_val_from_index(value, idx)[0]
                        )
                        self.vm.set(
                            name + "i", _get_val_from_index(value, idx)[1]
                        )

                else:
                    raise Exception(
                        "The shape of value should be ", self.shape
                    )
            else:
                if value.shape == ():

                    def func(name, idx, **kwargs):
                        self.vm.set(name, value)

                elif value.shape == tuple(self.shape):

                    def func(name, idx, **kwargs):
                        self.vm.set(name, _get_val_from_index(value, idx))

                else:
                    raise Exception(
                        "The shape of value should be ", self.shape
                    )

            _shape_func(func, self.shape, self.name, idx=[])

    def set_rho(self, rho, index=None):
        if self.cplx is not True:
            raise Exception("This method only supports complex Variable!")
        if index is not None:
            assert len(index) == len(self.shape)
            var_name = self.name
            for i in index:
                var_name += "_" + str(index[i])
            self.vm.set(var_name + "r", rho)
        else:
            rho = np.array(rho)
            if rho.shape == ():

                def func(name, idx, **kwargs):
                    self.vm.set(name + "r", rho)

            elif rho.shape == tuple(self.shape):

                def func(name, idx, **kwargs):
                    self.vm.set(name + "r", _get_val_from_index(rho, idx)[0])

            else:
                raise Exception("The shape of rho should be ", self.shape)
            _shape_func(func, self.shape, self.name, idx=[])

    def set_phi(self, phi, index=None):
        if self.cplx is not True:
            raise Exception("This method only supports complex Variable!")
        if index is not None:
            assert len(index) == len(self.shape)
            var_name = self.name
            for i in index:
                var_name += "_" + str(index[i])
            self.vm.set(var_name + "i", phi)
        else:
            phi = np.array(phi)
            if phi.shape == ():

                def func(name, idx, **kwargs):
                    self.vm.set(name + "i", phi)

            elif phi.shape == tuple(self.shape):

                def func(name, idx, **kwargs):
                    self.vm.set(name + "i", _get_val_from_index(phi, idx)[0])

            else:
                raise Exception("The shape of phi should be ", self.shape)
            _shape_func(func, self.shape, self.name, idx=[])

    @property
    def variables(self):
        """
        Names of the real variables contained in this Variable instance.

        :return: List of string.
        """
        return self.vm.var_head[self]

    def rename(self, new_name):
        """Rename this Variable."""

        def func(name, idx):
            vn = self.name + name
            new_vn = new_name + name
            self.vm.rename_var(vn, new_vn, cplx=self.cplx)
            self.vm.var_head[self].remove(vn)
            self.vm.var_head[self].append(new_vn)

        _shape_func(func, self.shape, "")
        self.name = new_name

    def fixed(self, value=None):
        """
        Fix this Variable. Note only non-shape real Variable supports this method.

        :param value: Real number. The fixed value
        """
        if not self.shape:
            if self.cp_effect:
                if value is None:
                    value = [None, None, None, None]
                else:
                    cplx_value = complex(value)
                    value = [cplx_value.real, 0.0, 0.0, 0.0]
                    self.vm.set_fix(self.name + "r", value[0])
                    self.vm.set_fix(self.name + "i", value[1])
                    self.vm.set_fix(self.name + "deltar", value[2])
                    self.vm.set_fix(self.name + "deltai", value[3])
            elif self.cplx:
                if value is None:
                    value = [None, None]
                else:
                    cplx_value = complex(value)
                    value = [cplx_value.real, cplx_value.imag]
                self.vm.set_fix(self.name + "r", value[0])
                self.vm.set_fix(self.name + "i", value[1])
            else:
                self.vm.set_fix(self.name, value)
        else:
            raise Exception("Only shape==() real var supports 'fixed' method.")

    def is_fixed(self):
        ret = True
        check = lambda x: not (x in self.vm.trainable_vars)
        if self.shape:
            ret = False
        else:
            if self.cp_effect:
                ret = ret and check(self.name + "r")
                ret = ret and check(self.name + "i")
                ret = ret and check(self.name + "deltar")
                ret = ret and check(self.name + "deltai")
            elif self.cplx:
                ret = ret and check(self.name + "r")
                ret = ret and check(self.name + "i")
            else:
                ret = ret and check(self.name)
        return ret

    def freed(self):
        """
        Set free this Variable. Note only non-shape Variable supports this method.
        """
        if not self.shape:
            if self.cp_effect:
                self.vm.set_fix(self.name + "r", unfix=True)
                self.vm.set_fix(self.name + "i", unfix=True)
                self.vm.set_fix(self.name + "deltar", unfix=True)
                self.vm.set_fix(self.name + "deltai", unfix=True)
            elif self.cplx:
                self.vm.set_fix(self.name + "r", unfix=True)
                self.vm.set_fix(self.name + "i", unfix=True)
            else:
                self.vm.set_fix(self.name, unfix=True)
        else:
            for i in self.variables:
                self.vm.set_fix(i, unfix=True)

    def _set_fix_idx(self, fix_idx=None, fix_vals=None, unfix=False):
        if fix_idx is None:
            fix_idx = []
        else:
            fix_idx = np.array(fix_idx)
            if len(fix_idx.shape) <= 1:
                fix_idx = np.reshape(fix_idx, (-1, 1))
            shape_mod = [self.shape[-i - 1] for i in range(fix_idx.shape[-1])]
            fix_idx = fix_idx % shape_mod
        fix_idx_str = ["_" + "_".join(str(j) for j in i) for i in fix_idx]

        if self.cp_effect:
            print(
                "I am cp_effect for set_fix_idx  fix_idx_str",
                fix_idx_str,
                " fix_vals ",
                fix_vals,
            )
            # print("I am cp_effect for set_fix_idx  free_idx_str", free_idx_str)
            if fix_vals is None:
                print("fix_vals is None", fix_vals)
                fix_vals = [None, None, None, None]
            elif not hasattr(fix_vals, "__len__"):
                fix_vals = [fix_vals, 0.0, fix_vals, 0.0]
            if len(fix_vals) < 4:
                fix_vals = (*fix_vals, 0, 0)
            print("fix_vals ", fix_vals)

            def func(name, idx):
                for ss in fix_idx_str:
                    if name.endswith(ss):
                        # print("set_fix_idx set name+r ", name)
                        self.vm.set_fix(
                            name + "r", value=fix_vals[0], unfix=unfix
                        )
                        self.vm.set_fix(
                            name + "i", value=fix_vals[1], unfix=unfix
                        )
                        if len(fix_vals) > 2:
                            self.vm.set_fix(
                                name + "deltar", value=fix_vals[2], unfix=unfix
                            )
                            self.vm.set_fix(
                                name + "deltai", value=fix_vals[3], unfix=unfix
                            )

            _shape_func(func, self.shape, self.name)

        elif self.cplx:
            if fix_vals is None:
                fix_vals = [None, None]
            elif not hasattr(fix_vals, "__len__"):
                fix_vals = [fix_vals, 0.0]

            def func(name, idx):
                for ss in fix_idx_str:
                    if name.endswith(ss):
                        self.vm.set_fix(
                            name + "r", value=fix_vals[0], unfix=unfix
                        )
                        self.vm.set_fix(
                            name + "i", value=fix_vals[1], unfix=unfix
                        )

            _shape_func(func, self.shape, self.name)
        else:

            def func(name, idx):
                for ss in fix_idx_str:
                    if name.endswith(ss):
                        self.vm.set_fix(name, value=fix_vals, unfix=unfix)

            _shape_func(func, self.shape, self.name)

    def set_fix_idx(self, fix_idx=None, fix_vals=None, free_idx=None):
        """
        :param fix_idx: Interger or list of integers. Which complex component in the innermost layer of the variable is fixed. E.g. If ``self.shape==[2,3,4]`` and ``fix_idx==[1,2]``, then Variable()[i][j][1] and Variable()[i][j][2] will be the fixed value.
        :param fix_vals: Float or length-2 float list for complex variable. The fixed value.
        :param free_idx: Interger or list of integers. Which complex component in the innermost layer of the variable is set free. E.g. If ``self.shape==[2,3,4]`` and ``fix_idx==[0]``, then Variable()[i][j][0] will be set free.
        """
        if not self.shape:
            raise Exception(
                "Only shape!=() var supports 'set_fix_idx' method to fix or free variables."
            )
        self._set_fix_idx(fix_idx, fix_vals)
        self._set_fix_idx(free_idx, unfix=True)

    def set_bound(self, bound, func=None, overwrite=False):
        """
        Set boundary for this Variable. Note only non-shape real Variable supports this method.

        :param bound: Length-2 tuple.
        :param func: String. Refer to class **tfpcbpggsz.variable.Bound**.
        :param overwrite: Boolean. If it's ``True``, the program will not throw a warning when overwrite a variable with the same name.
        """
        if not self.shape:
            self.vm.set_bound({self.name: bound}, func, overwrite=overwrite)
            self.bound = self.vm.bnd_dic[self.name]
        else:
            raise Exception(
                "Only shape==() real var supports 'set_bound' method."
            )

    def set_same_ratio(self):
        assert self.cplx, "variable should be complex"

        var = []

        def func(name, idx):
            var.append(name)

        _shape_func(func, self.shape, self.name)
        self.vm.set_share_r(var)

    def r_shareto(self, Var):
        """
        Share the radium component to another Variable of the same shape. Only complex Variable supports this method.

        :param Var: Variable.
        """
        if self.shape != Var.shape:
            raise Exception("Shapes are not the same.")
        if not (self.cplx and Var.cplx):
            raise Exception("Type is not complex var.")

        def func(name, idx, **kwargs):
            self.vm.set_share_r([self.name + name, Var.name + name])

        _shape_func(func, self.shape, "")

    def sameas(self, Var):
        """
        Set the Variable to be the same with another Variable of the same shape.

        :param Var: Variable.
        """
        if self.shape != Var.shape:
            raise Exception("Shapes are not the same.")
        if self.cplx != Var.cplx:
            raise Exception("Types (real or complex) are not the same.")

        def func(name, idx, **kwargs):
            self.vm.set_same(
                [self.name + name, Var.name + name], cplx=self.cplx
            )

        _shape_func(func, self.shape, "")

    def init_name_list(self):
        all_name_list = []
        if self.shape:

            def func(name, idx, **kwargs):
                nonlocal all_name_list
                if self.cp_effect:
                    all_name_list += [
                        name + "r",
                        name + "deltar",
                        name + "i",
                        name + "deltai",
                    ]
                elif self.cplx:
                    all_name_list += [name + "r", name + "i"]
                else:
                    all_name_list.append(name)

            _shape_func(func, self.shape, self.name)
        else:
            if self.cplx:
                name = self.name
                all_name_list += [name + "r", name + "i"]
            else:
                all_name_list.append(self.name)

        return all_name_list

    def factor_names(self):
        if self.cplx:
            return [i for i in self.all_name_list if i.endswith("r")]
        return self.all_name_list

    def __call__(self, charge=1):
        var = [self.vm.read(i) for i in self.all_name_list]
        if self.shape:
            if self.cp_effect:
                r = tf.stack(var[::4])
                deltar = tf.stack(var[1::4])
                i = tf.stack(var[2::4])
                deltai = tf.stack(var[3::4])
                cplx = []
                for k in self.all_name_list[::4]:
                    cond_i = self.vm.complex_vars.get(k[:-1], False)
                    cplx.append(cond_i)
                cond = tf.stack(cplx)
                real = r + charge * deltar
                imag = i + charge * deltai
                ret_polar = tf.complex(
                    real * tf.math.cos(imag), real * tf.math.sin(imag)
                )
                ret_rect = tf.complex(real, imag)
                ret = tf.where(cond, ret_polar, ret_rect)
            elif self.cplx:
                r = tf.stack(var[::2])
                i = tf.stack(var[1::2])
                cplx = []
                for k in self.all_name_list[::2]:
                    cond_i = self.vm.complex_vars.get(k[:-1], False)
                    cplx.append(cond_i)
                cond = tf.stack(cplx)
                ret_polar = tf.complex(r * tf.math.cos(i), r * tf.math.sin(i))
                ret_rect = tf.complex(r, i)
                ret = tf.where(cond, ret_polar, ret_rect)
            else:
                ret = tf.stack(var)

            return tf.reshape(ret, self.shape)
        else:
            if self.cplx:
                name = self.name
                if (name in self.vm.complex_vars) and self.vm.complex_vars[
                    name
                ]:
                    real = var[0] * tf.cos(var[1])
                    imag = var[0] * tf.sin(var[1])
                    var_list = tf.complex(real, imag)
                else:
                    var_list = tf.complex(var[0], var[1])
            else:
                var_list = var[0]

        # return tf.stack(var_list)
        return var_list


def deep_stack(dic, deep=1):
    if isinstance(dic, dict):
        return {k: v for k, v in dic.items()}
    elif isinstance(dic, list):
        flag = True
        tmp = dic
        for i in range(deep):
            if len(tmp) > 0 and isinstance(tmp, list):
                tmp = tmp[0]
            else:
                flag = False
                break
        if flag and isinstance(tmp, tf.Tensor):
            return tf.stack(dic)
        else:
            return [deep_stack(i, deep) for i in dic]
    elif isinstance(dic, tuple):
        return tuple([deep_stack(i, deep) for i in dic])
    return dic


class SumVar:
    def __init__(self, value, grad, var, hess=None):
        self.var = var
        self.value = tf.nest.map_structure(tf.stop_gradient, value)
        self.grad = grad
        self.hess = hess

    def from_call(fun, var, *args, **kwargs):
        with tf.GradientTape(persistent=True) as tape:
            y = tf.nest.map_structure(tf.reduce_sum, fun(*args, **kwargs))
        dy = tf.nest.map_structure(
            lambda x: tf.stack(
                tape.gradient(x, var, unconnected_gradients="zero")
            ),
            y,
        )
        del tape
        return SumVar(y, dy, var)

    def from_call_with_hess(fun, var, *args, **kwargs):
        with tf.GradientTape(persistent=True) as tape0:
            with tf.GradientTape(persistent=True) as tape:
                y = tf.nest.map_structure(tf.reduce_sum, fun(*args, **kwargs))
            dy = tf.nest.map_structure(
                lambda x: tape.gradient(x, var, unconnected_gradients="zero"),
                y,
            )
            del tape
        ddy = tf.nest.map_structure(
            lambda x: tf.stack(
                tape0.gradient(x, var, unconnected_gradients="zero")
            ),
            dy,
        )
        del tape0
        return SumVar(y, deep_stack(dy), var, hess=deep_stack(ddy))

    def __call__(self):
        var = tf.stack(self.var)
        if self.hess is None:
            fun = lambda x, y: x + tf.reduce_sum(
                y * (var - tf.stop_gradient(var))
            )
            return tf.nest.map_structure(fun, self.value, self.grad)
        else:

            def fun(x, y, z):
                delta = var - tf.stop_gradient(var)
                tmp = x + tf.reduce_sum(y * delta)
                tmp = tmp + 0.5 * tf.reduce_sum(
                    tf.reduce_sum(self.hess * delta, axis=-1) * delta
                )
                return tmp

            return tf.nest.map_structure(fun, self.value, self.grad, self.hess)

    def __add__(self, others):
        if not isinstance(others, SumVar):
            return NotImplemented
        value = tf.nest.map_structure(
            lambda x, y: x + y, self.value, others.value
        )
        grad = tf.nest.map_structure(
            lambda x, y: x + y, self.grad, others.grad
        )
        hess = None
        if self.hess is not None and others.hess is not None:
            hess = tf.nest.map_structure(
                lambda x, y: x + y, self.hess, others.hess
            )
        return SumVar(value, grad, self.var, hess=hess)
