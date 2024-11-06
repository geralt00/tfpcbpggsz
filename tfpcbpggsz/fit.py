import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import json
import time
from tfpcbpggsz.core import DecayNLLCalculator
from scipy.optimize import BFGS, basinhopping, minimize


class LargeNumberError(ValueError):
    pass


def fit_minuit(fcn, bounds_dict={}, hesse=True, minos=False, **kwargs):
    try:
        import iminuit
    except ImportError:
        raise RuntimeError(
            "You haven't installed iminuit so you can't use Minuit to fit."
        )
    """

    :param fcn:
    :param bounds_dict:
    :param hesse:
    :param minos:
    :return:
    """
    if int(iminuit.__version__[0]) < 2:
        return fit_minuit_v1(
            fcn, bounds_dict=bounds_dict, hesse=hesse, minos=minos, **kwargs
        )
    return fit_minuit_v2(
        fcn, bounds_dict=bounds_dict, hesse=hesse, minos=minos, **kwargs
    )

def fit_minuit_v1(fcn, bounds_dict={}, hesse=True, minos=False, **kwargs):

    """

    :param fcn:
    :param bounds_dict:
    :param hesse:
    :param minos:
    :return:
    """
    from iminuit import Minuit

    var_args = {}
    var_names = fcn.vm.trainable_vars
    for i in var_names:
        var_args[i] = fcn.vm.get(i)
        if i in bounds_dict:
            var_args["limit_{}".format(i)] = bounds_dict[i]
        var_args["error_" + i] = 0.1

    f_g = Cached_FG(fcn.nll_grad)
    m = Minuit(
        f_g.fun,
        name=var_names,
        errordef=0.5,
        grad=f_g.grad,
        print_level=2,
        use_array_call=True,
        **var_args,
    )
    print("########## begin MIGRAD")
    now = time.time()
    m.migrad()  # (ncall=10000))#,precision=5e-7))
    print("MIGRAD Time", time.time() - now)
    if hesse:
        print("########## begin HESSE")
        now = time.time()
        m.hesse()
        print("HESSE Time", time.time() - now)
    if minos:
        print("########## begin MINOS")
        now = time.time()
        m.minos()  # (var="")
        print("MINOS Time", time.time() - now)
    ndf = len(m.list_of_vary_param())
    ret = FitResult(
        dict(m.values), fcn, m.fval, ndf=ndf, success=m.migrad_ok()
    )
    ret.set_error(dict(m.errors))
    return ret


def fit_minuit_v2(fcn, bounds_dict={}, hesse=True, minos=False, **kwargs):

    """

    :param fcn:
    :param bounds_dict:
    :param hesse:
    :param minos:
    :return:
    """
    from iminuit import Minuit

    var_args = {}
    var_names = fcn.vm.trainable_vars
    x0 = []
    for i in var_names:
        x0.append(fcn.vm.get(i))
        var_args[i] = fcn.vm.get(i)
        if i in bounds_dict:
            var_args["limit_{}".format(i)] = bounds_dict[i]
        # var_args["error_" + i] = 0.1

    f_g = Cached_FG(fcn.nll_grad)
    m = Minuit(
        f_g.fun,
        np.array(x0),
        name=var_names,
        grad=f_g.grad,
    )
    m.strategy = 0
    for i in var_names:
        if i in bounds_dict:
            m.limits[i] = bounds_dict[i]
    m.errordef = 0.5
    m.print_level = 2
    print("########## begin MIGRAD")
    now = time.time()
    m.migrad()  # (ncall=10000))#,precision=5e-7))
    print("MIGRAD Time", time.time() - now)
    if hesse:
        print("########## begin HESSE")
        now = time.time()
        m.hesse()
        print("HESSE Time", time.time() - now)
    if minos:
        print("########## begin MINOS")
        now = time.time()
        m.minos()  # (var="")
        print("MINOS Time", time.time() - now)
    ndf = len(var_names)
    ret = FitResult(
        dict(zip(var_names, m.values)), fcn, m.fval, ndf=ndf, success=m.valid
    )
    # print(m.errors)
    ret.set_error(dict(zip(var_names, m.errors)))
    return ret

class FitResult(object):
    def __init__(
        self, params, fcn, min_nll, ndf=0, success=True, hess_inv=None
    ):
        self.params = params
        self.error = {}
        self.fcn = fcn
        self.min_nll = float(min_nll)
        self.ndf = int(ndf)
        self.success = success
        self.hess_inv = hess_inv
        self.extra = {}

    def save_as(self, file_name, save_hess=False):
        s = {
            "value": self.params,
            "error": self.error,
            "status": {
                "success": self.success,
                "NLL": self.min_nll,
                "Ndf": self.ndf,
                **self.extra,
            },
        }
        if save_hess and self.hess_inv is not None:
            s["free_params"] = [str(i) for i in self.error]
            s["hess_inv"] = [[float(j) for j in i] for i in self.hess_inv]
        with open(file_name, "w") as f:
            json.dump(s, f, indent=2)

    def set_error(self, error):
        self.error = error.copy()

