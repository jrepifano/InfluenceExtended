import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# from sklearn.learning_curve import learning_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve,auc, precision_recall_curve, average_precision_score


def save_plots(model_name, best_estimator, x_train_df, y_train_df, y_test_df, test_probabilities):

	"""
	Generate and Save:
		-Precision-Recall Curve
		-Calibration
		-ROC
		-Learning Curve

	Parameters
	----------
	model_name : string
		model name for saving files.

	best_estimator : object
		Best model output from sklearn

	x_train_df : training data dataframe
		Used for learning curve

	y_train_df : target values for training set dataframe
		Used for learning curve

	y_test_df : target values for test set dataframe
		Used for precision recall, calibration, and ROC

	test_probabilities : probability values for target class on test set
		Used for precision recall, calibration, and ROC
	"""

	#Plot Precision Recall Curve 
	average_precision = average_precision_score(y_test_df, test_probabilities)
	precision, recall, _ = precision_recall_curve(y_test_df, test_probabilities)
	plt.step(recall, precision, color='k', alpha=0.2, where='post')
	plt.fill_between(recall, precision, step='post', alpha=0.2,color='k')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title(model_name + ' Precision-Recall Curve: AP={0:0.2f}'.format(average_precision))
	plt.grid()

	#Save PR Curve
	pr_title = model_name + '_Precision_Recall_Curve.pdf'
	pdf = PdfPages(os.path.join(pr_title))
	pdf.savefig(dpi=600, bbox_inches='tight', pad_inches=.15)
	pdf.close()

	# Plot Calibration
	plt.figure(figsize=(10, 10))
	ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
	ax2 = plt.subplot2grid((3, 1), (2, 0))
	fraction_of_positives, mean_predicted_value = calibration_curve(y_test_df, test_probabilities, n_bins=7)
	ax1.plot(mean_predicted_value, fraction_of_positives, "s-",color='k')
	ax2.hist(test_probabilities, range=(0, 1), bins=7, histtype="step", lw=2,color='k')

	ax1.set_ylabel("Fraction of positives")
	ax1.set_ylim([-0.05, 1.05])
	ax1.legend(loc="lower right")
	ax1.set_title(model_name + ' Model Calibration')

	ax2.set_xlabel("Mean predicted value")
	ax2.set_ylabel("Count")
	ax2.legend(loc="upper center", ncol=2)

	plt.tight_layout()
	plt.grid()

	#Save Calibration
	calibration_title = model_name + '_Calibration.pdf'
	pdf = PdfPages(os.path.join(calibration_title))
	pdf.savefig(dpi=600, bbox_inches='tight', pad_inches=.15)
	pdf.close()
	plt.show()

	# Plot ROC
	fpr, tpr, thresholds = roc_curve(y_test_df, test_probabilities, pos_label=1)
	roc_auc=auc(fpr,tpr)
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='black',
		lw=lw, label='ROC curve(area = %0.2f)'%roc_auc)
	plt.plot([0, 1], [0, 1], color='darkgrey', lw=lw, linestyle='--', label='Random Guess')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate', fontsize=12)
	plt.ylabel('True Positive Rate',fontsize=12)
	plt.title(model_name + ' ROC',fontsize=14)
	plt.legend(loc="lower right")
	plt.grid()

	roc_title = model_name + '_ROC.pdf'
	pdf = PdfPages(os.path.join(roc_title))
	pdf.savefig(dpi=600, bbox_inches='tight', pad_inches=.15)
	pdf.close()
	plt.show

	#Plot Learning Curve
# 	ylim=None
# 	cv=5
# 	n_jobs = 1
# 	train_sizes=np.linspace(.1, 1.0, 5)

# 	plt.figure()
# 	plt.title(model_name+' Learning Curve')
# 	if ylim is not None:
# 		plt.ylim(*ylim)
# 	plt.xlabel("Training examples")
# 	plt.ylabel("Score (CV F1 Avg)")
# 	train_sizes, train_scores, test_scores = learning_curve(
# 		best_estimator, x_train_df, y_train_df, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,scoring='f1')
# 	train_scores_mean = np.mean(train_scores, axis=1)
# 	train_scores_std = np.std(train_scores, axis=1)
# 	test_scores_mean = np.mean(test_scores, axis=1)
# 	test_scores_std = np.std(test_scores, axis=1)
# 	plt.grid()

# 	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
# 					 train_scores_mean + train_scores_std, alpha=0.1,
# 					 color="k")
# 	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
# 					 test_scores_mean + test_scores_std, alpha=0.1, color="k")
# 	plt.plot(train_sizes, train_scores_mean, '+-', color="k",
# 			 label="Training score")
# 	plt.plot(train_sizes, test_scores_mean, 'o-', color="k",
# 			 label="Cross-validation score")

# 	plt.legend(loc="best")


# 	lc_title = model_name + '_Learning_curve.pdf'
# 	pdf = PdfPages(os.path.join(lc_title))
# 	pdf.savefig(dpi=600, bbox_inches='tight', pad_inches=.15)
# 	pdf.close()
# 	


	



