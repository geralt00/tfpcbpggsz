import abc
import time
from typing import Any


class BaseGenerator(metaclass=abc.ABCMeta):
    DataType = Any

    @abc.abstractmethod
    def generate(self, N: int) -> Any:
        raise NotImplementedError("generate")


class GenTest:
    def __init__(self, N_max, display=True):
        self.N_max = N_max
        self.N_gen = 0
        self.N_total = 0
        self.eff = 0.9
        self.display = display

    def generate(self, N):
        self.N_gen = 0
        self.N_total = 0

        N_progress = 50
        start_time = time.perf_counter()
        while self.N_gen < N:
            test_N = min(int((N - self.N_gen) / self.eff * 1.1), self.N_max)
            self.N_total += test_N
            yield test_N
            progress = self.N_gen / N + 1e-5
            finsh = "▓" * int(progress * N_progress)
            need_do = "-" * (N_progress - int(progress * N_progress) - 1)
            now = time.perf_counter() - start_time
            if self.display:
                print(
                    "\r{:^3.1f}%[{}>{}] {:.2f}/{:.2f}s eff: {:.6f}%  ".format(
                        progress * 100,
                        finsh,
                        need_do,
                        now,
                        now / progress,
                        self.eff * 100,
                    ),
                    end="",
                )
            self.eff = (self.N_gen + 1) / (self.N_total + 1)  # avoid zero
        end_time = time.perf_counter() - start_time
        if self.display:
            print(
                "\r{:^3.1f}%[{}] {:.2f}/{:.2f}s  eff: {:.6f}%   ".format(
                    100, "▓" * N_progress, end_time, end_time, self.eff * 100
                )
            )

    def add_gen(self, n_gen):
        # print("add gen")
        self.N_gen = self.N_gen + n_gen

    def set_gen(self, n_gen):
        # print("set gen")
        self.N_gen = n_gen


def multi_sampling(
    phsp,
    amp,
    N,
    max_N=200000,
    force=True,
    max_weight=None,
    importance_f=None,
    display=True,
):

    import tensorflow as tf

    from tfpcbpggsz.generator.data import data_mask, data_merge, data_shape

    a = GenTest(max_N, display=display)
    all_data = []

    for i in a.generate(N):
        data, new_max_weight = single_sampling2(
            phsp, amp, i, max_weight, importance_f
        )
        if max_weight is None:
            max_weight = new_max_weight * 1.1
        if new_max_weight > max_weight and len(all_data) > 0:
            tmp = data_merge(*all_data)
            rnd = tf.random.uniform((data_shape(tmp),), dtype=max_weight.dtype)
            cut = (
                rnd * new_max_weight / max_weight < 1.0
            )  # .max_amplitude < 1.0
            max_weight = new_max_weight * 1.05
            tmp = data_mask(tmp, cut)
            all_data = [tmp]
            a.set_gen(data_shape(tmp))
        a.add_gen(data_shape(data))
        # print(a.eff, a.N_gen, max_weight)
        all_data.append(data)

    ret = data_merge(*all_data)

    if force:
        cut = tf.range(data_shape(ret)) < N
        ret = data_mask(ret, cut)

    status = (a, max_weight)

    return ret, status


def multi_sampling2(
    phsp,
    amp,
    N,
    max_N=200000,
    force=True,
    max_weight=None,
    importance_f=None,
    display=True,
):

    import tensorflow as tf

    from tfpcbpggsz.generator.data import data_mask, data_merge, data_shape

    a = GenTest(max_N, display=display)
    all_data_sig = []
    all_data_tag = []

    for i in a.generate(N):
        data_sig, data_tag, new_max_weight = double_sampling2(
            phsp, amp, i, max_weight, importance_f
        )
        if max_weight is None:
            max_weight = new_max_weight * 1.1
        if new_max_weight > max_weight and len(all_data_sig) > 0 and len(all_data_tag) > 0:
            tmp_sig = data_merge(*all_data_sig)
            tmp_tag = data_merge(*all_data_tag)
            rnd = tf.random.uniform((data_shape(tmp_sig),), dtype=max_weight.dtype)
            cut = (
                rnd * new_max_weight / max_weight < 1.0
            )  # .max_amplitude < 1.0
            max_weight = new_max_weight * 1.05
            tmp_sig = data_mask(tmp_sig, cut)
            tmp_tag = data_mask(tmp_tag, cut)
            all_data_sig = [tmp_sig]
            all_data_tag = [tmp_tag]
            a.set_gen(data_shape(tmp_sig)+data_shape(tmp_tag))

        a.add_gen(data_shape(data_sig))
        a.add_gen(data_shape(data_tag))
        
        # print(a.eff, a.N_gen, max_weight)
        all_data_sig.append(data_sig)
        all_data_tag.append(data_tag)

    ret_sig = data_merge(*all_data_sig)
    ret_tag = data_merge(*all_data_tag)

    if force:
        cut = tf.range(data_shape(ret_sig)) < N
        ret_sig = data_mask(ret_sig, cut)
        ret_tag = data_mask(ret_tag, cut)

    status = (a, max_weight)

    return ret_sig, ret_tag, status

def double_sampling2(phsp, amp, N, max_weight=None, importance_f=None):
    """
    Double sampling based on correlation between two decays
    """
    import tensorflow as tf

    from tfpcbpggsz.generator.data import data_mask

    time1 = time.time()
    data_sig = phsp(N)
    data_tag = phsp(N)
    time2 = time.time()
    weight = amp(data_sig, data_tag)
    time3 = time.time()
    #Not yet sure about the importance_f
    if importance_f is not None:
        weight = weight / importance_f(data_sig, data_tag)

    new_max_weight = tf.reduce_max(weight)
    if max_weight is None or max_weight < new_max_weight:
        max_weight = new_max_weight * 1.01
    time4 = time.time()
    rnd = tf.random.uniform(weight.shape, dtype=weight.dtype)
    time5 = time.time()
    cut = rnd * max_weight < weight
    time6 = time.time()
    data_sig = data_mask(data_sig, cut)
    data_tag = data_mask(data_tag, cut)
    time7 = time.time()

    #print(f"Time taken for phsp: {time2-time1}")
    #print(f"Time taken for amp: {time3-time2}")
    #print(f"Time taken for importance_f: {time4-time3}")
    #print(f"Time taken for rnd: {time5-time4}")
    #print(f"Time taken for cut: {time6-time5}")
    #print(f"Time taken for data_mask: {time7-time6}")
    #print(f"Total time taken: {time7-time1}")
    return data_sig, data_tag, max_weight


def single_sampling2(phsp, amp, N, max_weight=None, importance_f=None):
    import tensorflow as tf

    from tfpcbpggsz.generator.data import data_mask

    data = phsp(N)
    time1 = time.time()
    weight = amp(data)
    if weight.numpy().any() < 0.0:
        print("Negative weight found")
        
    time2 = time.time()
    if importance_f is not None:
        weight = weight / importance_f(data)
    new_max_weight = tf.reduce_max(weight)
    if max_weight is None or max_weight < new_max_weight:
        max_weight = new_max_weight * 1.01
    rnd = tf.random.uniform(weight.shape, dtype=weight.dtype)
    cut = rnd * max_weight < weight
    data = data_mask(data, cut)
    #print(f"Time taken for amp: {time2-time1}")
    return data, max_weight


class ARGenerator(BaseGenerator):
    """Acceptance-Rejection Sampling"""

    def __init__(self, phsp, amp, max_weight=None):
        self.phsp = phsp
        self.amp = amp
        self.max_weight = max_weight
        self.status = None

    def generate(self, N):
        ret, status = multi_sampling(
            self.phsp, self.amp, N, max_weight=self.max_weight, display=False
        )
        self.status = status
        return ret