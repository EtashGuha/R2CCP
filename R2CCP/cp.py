import torch
import numpy as np
import pickle 
import os

def percentile_excluding_index(vector, percentile):
        percentile_value = torch.quantile(vector, percentile)
        
        return percentile_value

def find_intervals_above_value_with_interpolation(x_values, y_values, cutoff):
    intervals = []
    start_x = None
    if y_values[0] >= cutoff:
        start_x = x_values[0]
    for i in range(len(x_values) - 1):
        x1, x2 = x_values[i], x_values[i + 1]
        y1, y2 = y_values[i], y_values[i + 1]

        if min(y1, y2) <= cutoff < max(y1, y2):
            # Calculate the x-coordinate where the line crosses the cutoff value
            x_cross = x1 + (x2 - x1) * (cutoff - y1) / (y2 - y1)

            if x1 <= x_cross <= x2:
                if start_x is None:
                    start_x = x_cross
                else:
                    intervals.append((start_x, x_cross))
                    start_x = None

    # If the line ends above cutoff, add the last interval
    if start_x is not None:
        intervals.append((start_x, x_values[-1]))

    return intervals

    
def get_all_scores(range_vals, X, y, model):
    step_val = (max(range_vals) - min(range_vals))/(len(range_vals) - 1)
    indices_up = torch.ceil((y - min(range_vals))/step_val).squeeze()
    indices_down = torch.floor((y - min(range_vals))/step_val).squeeze()
    
    how_much_each_direction = ((y.squeeze() - min(range_vals))/step_val - indices_down)

    weight_up = how_much_each_direction
    weight_down = 1 - how_much_each_direction

    bad_indices = torch.where(torch.logical_or(y.squeeze() > max(range_vals), y.squeeze() < min(range_vals)))
    indices_up[bad_indices] = 0
    indices_down[bad_indices] = 0
    
    scores = get_scores(X, model)
    all_scores = scores[torch.arange(len(X)), indices_up.long()] * weight_up + scores[torch.arange(len(X)), indices_down.long()] * weight_down
    all_scores[bad_indices] = 0
    return scores, all_scores

def get_scores(X, model):
    scores = torch.nn.functional.softmax(model(torch.tensor(X, dtype=torch.float32)), dim=1)
    return scores

def get_predictions(X, model, range_vals):
    pred_scores = get_scores(X, model)
    all_vals = []
    best_indices = torch.argmax(pred_scores, dim=1)
    all_vals = range_vals[best_indices]
    return all_vals

def get_cp_lists(X, args, range_vals, X_cal, y_cal, model):
    scores, all_scores = get_all_scores(range_vals, X_cal, y_cal, model)
    pred_scores = get_scores(X, model)
    alpha = args.alpha

    percentile_val = percentile_excluding_index(all_scores, alpha)
        
    all_intervals = []
    for i in range(len(pred_scores)):
        all_intervals.append(find_intervals_above_value_with_interpolation(range_vals, pred_scores[i], percentile_val))

    return all_intervals

def calc_coverages_and_lengths(all_intervals, y):
    coverages = []
    lengths = []
    for idx, intervals in enumerate(all_intervals):
        if len(intervals) == 0:
            length = 0
            cov_val = 0
        else:
            length = 0
            cov_val = 0
            for interval in intervals:
                length += interval[1] - interval[0]
                if interval[1]  >= y[idx].item() and y[idx].item() >= interval[0]:
                    cov_val = 1
        coverages.append(cov_val)
        lengths.append(length)

    return coverages, lengths

def calc_lengths(all_intervals):
    lengths = []
    for idx, intervals in enumerate(all_intervals):
        if len(intervals) == 0:
            length = 0
            cov_val = 0
        else:
            length = 0
            cov_val = 0
            for interval in intervals:
                length += interval[1] - interval[0]
        lengths.append(length)

    return lengths