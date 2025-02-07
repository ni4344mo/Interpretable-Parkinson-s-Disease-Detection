import shap
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import colors


def shap_sum(x, model, dataname):
    shap_values = shap.TreeExplainer(model).shap_values(x)
    shap.summary_plot(shap_values, x, show=False)

    filename = "Shap_sum" + dataname + ".png"
    filename2 = "Shap_sum" + dataname + ".pdf"
    plt.savefig(filename, bbox_inches='tight')
    plt.savefig(filename2, bbox_inches='tight')
    plt.show()
    return shap_values


def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step: np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red', 'green', 'blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1],), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)


def shap_dep(x, features, shap_values, dataname):
    inv = cmap_map(lambda i: 1 - i, matplotlib.cm.BrBG)

    divnorm = colors.TwoSlopeNorm(vmin=-4.5, vcenter=0., vmax=2)
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 2, 1)
    variable_name = features[0]
    index_in_X = x.columns.get_loc(variable_name)
    Shap_variable = shap_values[:, index_in_X]
    plt.scatter(x[variable_name], Shap_variable, c=Shap_variable, norm=divnorm, cmap='bwr')
    # plt.xlabel(variable_name, fontsize=14)
    plt.xlabel(' ', fontsize=14)
    plt.ylabel('SHAP value', fontsize=17)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=20)
    plt.xlim(None, None)

    divnorm = colors.TwoSlopeNorm(vmin=-2, vcenter=0., vmax=4.5)
    plt.subplot(3, 2, 2)
    variable_name = features[1]
    index_in_X = x.columns.get_loc(variable_name)
    Shap_variable = shap_values[:, index_in_X]
    plt.scatter(x[variable_name], Shap_variable, c=Shap_variable, norm=divnorm, cmap='bwr')
    # plt.xlabel(variable_name, fontsize=14)
    plt.xlabel(' ', fontsize=14)
    plt.ylabel('SHAP value', fontsize=17)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=20)
    plt.xlim(None, None)

    divnorm = colors.TwoSlopeNorm(vmin=-2, vcenter=0., vmax=1.5)
    plt.subplot(3, 2, 3)
    variable_name = features[2]
    index_in_X = x.columns.get_loc(variable_name)
    Shap_variable = shap_values[:, index_in_X]
    plt.scatter(x[variable_name], Shap_variable, c=Shap_variable, norm=divnorm, cmap='bwr')
    # plt.xlabel(variable_name, fontsize=14)
    plt.xlabel(' ', fontsize=14)
    plt.ylabel('SHAP value', fontsize=17)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=20)
    plt.xlim(None, None)

    divnorm = colors.TwoSlopeNorm(vmin=-4, vcenter=0., vmax=2)
    plt.subplot(3, 2, 4)
    variable_name = features[3]
    index_in_X = x.columns.get_loc(variable_name)
    Shap_variable = shap_values[:, index_in_X]
    plt.scatter(x[variable_name], Shap_variable, c=Shap_variable, norm=divnorm, cmap='bwr')
    # plt.xlabel(variable_name, fontsize=14)
    plt.xlabel(' ', fontsize=14)
    plt.ylabel('SHAP value', fontsize=17)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=20)
    plt.xlim(None, None)

    divnorm = colors.TwoSlopeNorm(vmin=-2, vcenter=0., vmax=1.5)
    plt.subplot(3, 2, 5)
    variable_name = features[4]
    index_in_X = x.columns.get_loc(variable_name)
    Shap_variable = shap_values[:, index_in_X]
    plt.scatter(x[variable_name], Shap_variable, c=Shap_variable, norm=divnorm, cmap='bwr')
    # plt.xlabel(variable_name, fontsize=14)
    plt.xlabel(' ', fontsize=14)
    plt.ylabel('SHAP value', fontsize=17)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=20)
    # plt.xticks([0, 5, 10, 15])
    plt.xlim(None, None)

    divnorm = colors.TwoSlopeNorm(vmin=-2.5, vcenter=0., vmax=1.5)
    plt.subplot(3, 2, 6)
    variable_name = features[5]
    index_in_X = x.columns.get_loc(variable_name)
    Shap_variable = shap_values[:, index_in_X]
    plt.scatter(x[variable_name], Shap_variable, c=Shap_variable, norm=divnorm, cmap='bwr')
    # plt.xlabel(variable_name, fontsize=14)
    plt.xlabel(' ', fontsize=14)
    plt.ylabel('SHAP value', fontsize=17)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=20)
    plt.xlim(None, None)

    plt.tight_layout()
    filename = "Shap_dep" + dataname + ".png"
    # Show the plots
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
