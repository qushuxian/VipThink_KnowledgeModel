# 原始代码结构
import numpy as np
from pyBKT.fit import E_step
from pyBKT.fit import predict_onestep_states


def run(model, data, parallel=True):
    num_subparts = data["data"].shape[0]
    num_resources = len(model["learns"])

    result = E_step.run(data, model, 1, int(parallel))
    for j in range(num_resources):
        result['all_trans_softcounts'][j] = result['all_trans_softcounts'][j].transpose()
    for j in range(num_subparts):
        result['all_emission_softcounts'][j] = result['all_emission_softcounts'][j].transpose()

    state_predictions = predict_onestep_states.run(data, model, result['alpha'], int(parallel))

    p = state_predictions.shape

    state_predictions = state_predictions.flatten(order='C').reshape(p, order='F')
    correct_emission_predictions = \
        np.expand_dims(model["guesses"], axis=1) \
        @ np.expand_dims(state_predictions[0, :], axis=0) \
        + np.expand_dims(1 - model["slips"], axis=1) \
        @ np.expand_dims(state_predictions[1, :], axis=0)

    return correct_emission_predictions[0].ravel(), state_predictions[1]
