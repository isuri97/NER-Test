import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from classification.common.denial_config import args
from classification.util.label_converter import decode

from classification import ClassificationModel



#read data file and label each sentence as holocaust denial or either confirm
from classification.util.print_stat import print_information

confirm_data = pd.read_csv('/home/isuri/PycharmProjects/NER-Test/data/confirm.txt',sep='delimiter', header=None)
confirm_data = confirm_data.rename(columns = {0:'text'})
confirm_data['label'] = 1

denial_data= pd.read_csv('/home/isuri/PycharmProjects/NER-Test/data/denial.txt',sep='delimiter', header=None)
denial_data=denial_data.rename(columns = {0:'text'})
denial_data['label'] = 0

data = confirm_data.append(denial_data, ignore_index=True)
data.to_csv('new.csv')


print(data.columns)

X_train, X_test = train_test_split(data, test_size=0.1)

X_train.columns = ["text", "label"]
X_test.columns = ["text", "label"]

# define hyperparameter
train_args ={"reprocess_input_data": True,
             "fp16":False,
             "use_cuda":False,
             "num_train_epochs": 4}

model = ClassificationModel(
    "bert", "bert-base-cased", use_cuda=False, args={'fp16': False, 'num_train_epochs': 1, 'learning_rate': 1e-5}
)


# Train the model
model.train_model(X_train)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(X_test)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Sam was a Wizard"])




# # # Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
#
# train_data=[X_train, y_train]
# test_data=[X_test, y_test]

# train_df = pd.DataFrame(train_data)
# print(train_df.columns)
# # train_df.columns = ["text", "label"]




# #
# train_df = pd.DataFrame(train_data)
# test_df = pd.DataFrame(test_data)
#
# print(train_df)
#
#
# # train_data = [
# #     ["Example sentence belonging to class 1", 1],
# #     ["Example sentence belonging to class 0", 0],
# # ]
# # train_df = pd.DataFrame(train_data)
# #
# # eval_data = [
# #     ["Example eval sentence belonging to class 1", 1],
# #     ["Example eval sentence belonging to class 0", 0],
# # ]
# # eval_df = pd.DataFrame(eval_data)
# #
# # Create a ClassificationModel
#
# Create a ClassificationModel
model = ClassificationModel("roberta", "roberta-base")

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(test_df)

print(result)
#
#
#
#
#
# #
# #
# # test_sentences = X_test.tolist()
# # test_preds = np.zeros((len(test_df), args["n_fold"]))
#
#
#
# # for i in range(args["n_fold"]):
# #     olid_train, olid_validation = train_test_split(train_df, test_size=0.1, random_state=args["manual_seed"])
# #     model = ClassificationModel("roberta", "roberta-base", args=args,  eval_df=olid_validation)
# #     model.train_model(olid_train)
# #     print("Finished Training")
# #     model = ClassificationModel(model_type_or_path=args["best_model_dir"])
# #     predictions, raw_outputs = model.predict(test_sentences)
# #     test_preds[:, i] = predictions
# #     print("Completed Fold {}".format(i))
# #
# # final_predictions = []
# # for row in test_preds:
# #     row = row.tolist()
# #     final_predictions.append(int(max(set(row), key=row.count)))
# #
# # test_df['predictions'] = final_predictions
# # test_df['predictions'] = decode(test_df['predictions'])
# # test_df['labels'] = decode(test_df['labels'])
# #
# # print_information(test_df, "predictions", "labels")
# # # olid_test
# # model.train_model()
# # print("Finished Training")
# # # model = OffensiveNNModel(model_type_or_path=args["best_model_dir"])
# predictions, raw_outputs = model.predict(eval_df.text)
# test_preds[:, i] = predictions
#     print("Completed Fold {}".format(i))
#
# final_predictions = []
# for row in test_preds:
#     row = row.tolist()
#     final_predictions.append(int(max(set(row), key=row.count)))
#
# olid_test['predictions'] = final_predictions
# olid_test['predictions'] = decode(olid_test['predictions'])
# olid_test['labels'] = decode(olid_test['labels'])
#
# print_information(olid_test, "predictions", "labels")
#
#
#
#
#
# # Train the model
# model.train_model(train_df)
#
#
#
# print_information(eval_df, "predictions", "labels")
#

# Evaluate the model
# result, model_outputs, wrong_predictions = model.eval_model(eval_df)



# print_information(olid_test, "predictions", "labels")