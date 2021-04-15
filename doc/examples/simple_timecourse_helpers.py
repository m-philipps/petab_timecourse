from typing import Sequence, Tuple

from petab_timecourse import Timecourse

#p_ = 1
#timecourse_q_ = np.array((
##   ( t, q_),  # `t` is time, `q_` is the value that the parameter takes.
#    ( 0,  1),
#    (10,  0),
#    (20, -1),
#    (30,  0),
#    (40,  1),
#    (50,  None),  # None indicates the end of the simulation.
#))

translate_condition = {
    'q_positive': 1,
    'q_zero': 0,
    'q_negative': -1,
}

def get_analytical_x_(
        t: float,
        timecourse: Timecourse,
        #timecourse_q_: Sequence[Tuple[float, float]] = Timecourse,
        p_: float,
) -> float:
    """The expected value for $x_$.

    Parameters
    ----------
    t:
        The current time.
    timecourse_q_:
        The values that `q_` changes to as time progresses, as a sequence of
        tuples. In each tuple, the first value is a timepoint, and the second
        value is the value that `q_` changes to at that timepoint. A `q_` value
        of `None` indicates the final timepoint.
    """
    timepoints = timecourse.timepoints
    q_conditions = [translate_condition[condition] for condition in timecourse.condition_ids]
    timecourse_q_ = list(zip(timepoints, q_conditions))
    x_ = 0
    for row_index, (t_start, q_) in enumerate(timecourse_q_):
        # FIXME remove None
        if q_ is None or row_index == len(timecourse_q_) - 1:
            #previous_q_ = timecourse_q_[row_index-1][1]
            #x_ += p_ * previous_q_ * (t - t_start)
            x_ += p_ * q_ * (t - t_start)
            break
        t_end = timecourse_q_[row_index+1][0]
        if t <= t_end:
            x_ += p_ * q_ * (t - t_start)
            break
        x_ += p_ * q_ * (t_end - t_start)
    return x_


def get_analytical_sx_(
        t: float,
        timecourse: Timecourse,
        #timecourse_q_: Sequence[Tuple[float, float]] = Timecourse,
) -> float:
    """The expected sensitivity for x_ w.r.t. p_.

    Parameters
    ----------
    t:
        The current time.
    timecourse_q_:
        The values that `q_` changes to as time progresses, as a sequence of
        tuples. In each tuple, the first value is a timepoint, and the second
        value is the value that `q_` changes to at that timepoint. A `q_` value
        of `None` indicates the final timepoint.
    """
    timepoints = timecourse.timepoints
    q_conditions = [translate_condition[condition] for condition in timecourse.condition_ids]
    timecourse_q_ = list(zip(timepoints, q_conditions))
    sx_ = 0
    for row_index, (t_start, q_) in enumerate(timecourse_q_):
        # FIXME remove None
        if q_ is None or row_index == len(timecourse_q_) - 1:
            #previous_q_ = timecourse_q_[row_index-1][1]
            #sx_ += previous_q_ * (t - t_start)
            sx_ += q_ * (t - t_start)
            break
        t_end = timecourse_q_[row_index+1][0]
        if t <= t_end:
            sx_ += q_ * (t - t_start)
            break
        sx_ += q_ * (t_end - t_start)
    return sx_
