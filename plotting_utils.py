import numpy as np

def plot_errors(ax, dt_errs_stats, dt_erm_errs_stats, rf_errs_stats,
                gb_errs_stats, xgb_errs_stats, groups, dataset, legend=True, title=True, ylabel=True):
        bar_width = 0.1
        num_groups = len(groups)
        index = np.arange(num_groups)
        for g in range(num_groups):
                if g == 0:
                        ax.bar(g - 2.5 * bar_width, dt_errs_stats[g][0], bar_width, yerr=dt_errs_stats[g][1], capsize=3, label="Decision Tree (group ERM)",
                        color="blue")
                        ax.bar(g - 1.5 * bar_width, dt_erm_errs_stats[g][0], bar_width, yerr=dt_erm_errs_stats[g][1], capsize=3, label="Decision Tree (ERM)", color="salmon")
                        ax.bar(g - 0.5 * bar_width, rf_errs_stats[g][0], bar_width, yerr=rf_errs_stats[g][1], capsize=3, label="Random Forest (ERM)", color="green")
                        ax.bar(g + 0.5 * bar_width, gb_errs_stats[g][0], bar_width, yerr=gb_errs_stats[g][1], capsize=3, label="Gradient Boosting (ERM)", color="purple")
                        ax.bar(g + 1.5 * bar_width, xgb_errs_stats[g][0], bar_width, yerr=xgb_errs_stats[g][1], capsize=3, label="XGBoost (ERM)", color="orange")
                else:
                        ax.bar(g - 2.5 * bar_width, dt_errs_stats[g][0], bar_width, yerr=dt_errs_stats[g][1], capsize=3, color="blue")
                        ax.bar(g - 1.5 * bar_width, dt_erm_errs_stats[g][0], bar_width, yerr=dt_erm_errs_stats[g][1], capsize=3, color="salmon")
                        ax.bar(g - 0.5 * bar_width, rf_errs_stats[g][0], bar_width, yerr=rf_errs_stats[g][1], capsize=3, color="green")
                        ax.bar(g + 0.5 * bar_width, gb_errs_stats[g][0], bar_width, yerr=gb_errs_stats[g][1], capsize=3, color="purple")
                        ax.bar(g + 1.5 * bar_width, xgb_errs_stats[g][0], bar_width, yerr=xgb_errs_stats[g][1], capsize=3, color="orange")
        ax.set_ylabel('Group-conditional Error Rate')
        if title:
                ax.set_title('Group-conditional Error Rates for Decision Tree vs. Ensemble Methods ({})'.format(dataset))

        xticks = ['{}'.format(g) for g in groups]
        ax.set_xticks(index, xticks)
        if legend:
                ax.legend()

def plot_adult_agreements(ax, group_pairs_agreements, group_pairs, group_names, model_class, xticks, bar_width=0.2, bar_groups=4, legend=True):
    # Plot "agreement" for each pair of intersecting groups
    index = np.arange(bar_groups)
    num_group_pairs = len(group_pairs_agreements)

    for i in range(0, num_group_pairs, 3):
            ax.bar(i/(bar_groups - 1) - bar_width, 
                    group_pairs_agreements[i][0], bar_width,
                    yerr=group_pairs_agreements[i][1],
                    label="G{} ({}) $\cap$ G{} ({})".format(group_pairs[i][0], group_names[group_pairs[i][0]], group_pairs[i][1], group_names[group_pairs[i][1]]))
            ax.bar(i/(bar_groups - 1), 
                    group_pairs_agreements[i+1][0], bar_width,
                    yerr=group_pairs_agreements[i+1][1],
                    label="G{} ({}) $\cap$ G{} ({})".format(group_pairs[i+1][0], group_names[group_pairs[i+1][0]],
                    group_pairs[i+1][1], 
                    group_names[group_pairs[i+1][1]]))
            ax.bar(i/(bar_groups - 1) + bar_width, 
                    group_pairs_agreements[i+2][0], bar_width,
                    yerr=group_pairs_agreements[i+2][1],
                    label="G{} ({}) $\cap$ G{} ({})".format(group_pairs[i+2][0], group_names[group_pairs[i+2][0]], group_pairs[i+2][1], group_names[group_pairs[i+2][1]]))

    ax.set_xlabel('Intersecting Group')
    ax.set_ylabel('Agreement')
    ax.set_title('Group Agreements ({})'.format(model_class))
    ax.set_xticks(index, xticks)
    if legend:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                    fancybox=True, shadow=True, ncol=3)

def plot_adult_errors(ax, test_err, std_errs, erm_test_err, erm_std_errs, model_class, num_groups, group_names, bar_width=0.2):
    # Error rate plots
    index = np.arange(num_groups)

    for g in range(num_groups):
            if g == 0:
                    ax.bar(g - 0.5 * bar_width, test_err[g], bar_width, yerr=std_errs[g], capsize=3, label="Group-ERM", color="blue")
                    ax.bar(g + 0.5 * bar_width, erm_test_err[g], bar_width, yerr=erm_std_errs[g], capsize=3, label="ERM", color="orange")
            else:
                    ax.bar(g - 0.5 * bar_width, test_err[g], bar_width, yerr=std_errs[g], capsize=3, color="blue")
                    ax.bar(g + 0.5 * bar_width, erm_test_err[g], bar_width, yerr=erm_std_errs[g], capsize=3, color="orange")
            ax.set_ylabel('Group-conditional Error Rate')
            ax.set_title('Group-conditional Error Rates of ERM (ALL) vs. Group-ERM ({})'.format(model_class))

    xticks = ['G{} ({})'.format(g, group_names[g]) for g in range(num_groups)]
    ax.set_xticks(index, xticks)
    ax.set_ylim([0, 0.5])
    ax.legend()

XTICKS_RACE = ('G1 (R1)', 'G2 (R2)', 'G3 (R3)', 'G4 (R6)', 'G5 (R7)', 'G6 (R8)', 'G7 (R9)')
def plot_race_agreements(ax, group_pairs_agreements, group_pairs, group_names, title, bar_width=0.3, bar_groups=7, xticks=XTICKS_RACE, legend=True):
    index = np.arange(bar_groups)
    num_group_pairs = len(group_pairs_agreements)

    for i in range(num_group_pairs):
        if i % 2 == 0:
            ax.bar(int(i/2) - bar_width/2, 
                   group_pairs_agreements[i][0], 
                   bar_width,
                   yerr=group_pairs_agreements[i][1],
                   label="G{} ({}) $\cap$ G{} ({})".format(group_pairs[i][0], group_names[group_pairs[i][0]], group_pairs[i][1], group_names[group_pairs[i][1]]))
        elif i % 2 == 1:
            ax.bar(int(i/2) + bar_width/2, 
                   group_pairs_agreements[i][0],
                    bar_width,
                    yerr=group_pairs_agreements[i][1],
                    label="G{} ({}) $\cap$ G{} ({})".format(group_pairs[i][0], group_names[group_pairs[i][0]], group_pairs[i][1], group_names[group_pairs[i][1]]))

    ax.set_xlabel('Intersecting Group')
    ax.set_ylabel('Agreement')
    ax.set_title(title)
    ax.set_xticks(index, xticks)
    if legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                fancybox=True, shadow=True, ncol=4)
        
XTICKS_SEX_AGE = ('(M, Y)', '(M, O)', '(F, Y)', '(F, O)')
def plot_sex_age_agreements(ax, group_pairs_agreements, group_pairs, group_names, title, bar_width=0.3, bar_groups=4, xticks=XTICKS_SEX_AGE, legend=True):
    index = np.arange(bar_groups)
    ax.bar(0, group_pairs_agreements[0][0], 
           bar_width,
           yerr=group_pairs_agreements[0][1], 
           label="G{} ({}) $\cap$ G{} ({})".format(group_pairs[0][0], group_names[group_pairs[0][0]], group_pairs[0][1], group_names[group_pairs[0][1]]))
    ax.bar(1, group_pairs_agreements[1][0], 
           bar_width,
           yerr=group_pairs_agreements[1][1], 
           label="G{} ({}) $\cap$ G{} ({})".format(group_pairs[1][0], group_names[group_pairs[1][0]], group_pairs[1][1], group_names[group_pairs[1][1]]))
    ax.bar(2, group_pairs_agreements[2][0], 
           bar_width,
           yerr=group_pairs_agreements[2][1], 
           label="G{} ({}) $\cap$ G{} ({})".format(group_pairs[2][0], group_names[group_pairs[2][0]], group_pairs[2][1], group_names[group_pairs[2][1]]))
    ax.bar(3, group_pairs_agreements[3][0],
           bar_width, 
           yerr=group_pairs_agreements[3][1],
           label="G{} ({}) $\cap$ G{} ({})".format(group_pairs[3][0], group_names[group_pairs[3][0]],group_pairs[3][1], group_names[group_pairs[3][1]]))

    ax.set_xlabel('Intersecting Group')
    ax.set_ylabel('Agreement')
    ax.set_title(title)
    ax.set_xticks(index, xticks)
    if legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, shadow=True, ncol=2)