import numpy as np
import collections
import torch
from textwrap import indent
import json

#Se carga las 3 predictions para cada template ----------------
simple_1=np.loadtxt("/gaueko0/users/dromero012/Fine-Tuning_Tests/Version_1/Experiments/Twitter_Complaints/predictions_logits_test_Simple_1.txt")
simple_2=np.loadtxt("/gaueko0/users/dromero012/Fine-Tuning_Tests/Version_1/Experiments/Twitter_Complaints/predictions_logits_test_Simple_2.txt")
simple_3=np.loadtxt("/gaueko0/users/dromero012/Fine-Tuning_Tests/Version_1/Experiments/Twitter_Complaints/predictions_logits_test_Simple_3.txt")

print("Predictions_Simple----------","\n")
predictions_simple_1 = list(np.argmax(simple_1, axis=1))
predictions_simple_2 = list(np.argmax(simple_2, axis=1))
predictions_simple_3 = list(np.argmax(simple_3, axis=1))
count_0 = predictions_simple_1.count(0)
count_1 = predictions_simple_1.count(1)
print('Simple 1 Count Not:', count_0)
print('Simple 1 Count Related:', count_1)
print("\n")
count_0 = predictions_simple_2.count(0)
count_1 = predictions_simple_2.count(1)
print('Simple 2 Count Not :', count_0)
print('Simple 2 Count Related:', count_1)
print("\n")
count_0 = predictions_simple_3.count(0)
count_1 = predictions_simple_3.count(1)
print('Simple 3 Count Not:', count_0)
print('Simple 3 Count Related:', count_1)
print("\n")

print("Predictions_Complex--------------","\n")

complex_1=np.loadtxt("/gaueko0/users/dromero012/Fine-Tuning_Tests/Version_1/Experiments/Twitter_Complaints/predictions_logits_test_Complex_1.txt")
complex_2=np.loadtxt("/gaueko0/users/dromero012/Fine-Tuning_Tests/Version_1/Experiments/Twitter_Complaints/predictions_logits_test_Complex_2.txt")
complex_3=np.loadtxt("/gaueko0/users/dromero012/Fine-Tuning_Tests/Version_1/Experiments/Twitter_Complaints/predictions_logits_test_Complex_3.txt")

predictions_complex_1 = list(np.argmax(complex_1, axis=1))
predictions_complex_2 = list(np.argmax(complex_2, axis=1))
predictions_complex_3 = list(np.argmax(complex_3, axis=1))
count_0 = predictions_complex_1.count(0)
count_1 = predictions_complex_1.count(1)
print('Complex 1 Count Not:', count_0)
print('Complex 1 Count Related:', count_1)
print("\n")
count_0 = predictions_complex_2.count(0)
count_1 = predictions_complex_2.count(1)
print('Complex 2 Count Not:', count_0)
print('Complex 2 Count Related:', count_1)
print("\n")
count_0 = predictions_complex_3.count(0)
count_1 = predictions_complex_3.count(1)
print('Complex 3 Count Not:', count_0)
print('Complex 3 Count Related:', count_1)
print("\n")

#Se hace un solo archivo - con promedio -----------------
comb_simple=(simple_1+simple_2+simple_3)/3
print("Simple Templates Combined-------------","\n")
print(comb_simple,"\n")

predictions_simple = list(np.argmax(comb_simple, axis=1))

print("Predictions","\n")
count_0 = predictions_simple.count(0)
count_1 = predictions_simple.count(1)
print('Comb Simple Count Not:', count_0)
print('Comb Simple Count Related:', count_1)
print("\n")

comb_complex=(complex_1+complex_2+complex_3)/3
print("Complex Templates Combined----------","\n")
print(comb_complex,"\n")

predictions_complex = list(np.argmax(comb_complex, axis=1))

print("Predictions","\n")
count_0 = predictions_complex.count(0)
count_1 = predictions_complex.count(1)
print('Comb Complex Count Not:', count_0)
print('Comb Complex Count Related:', count_1)
print("\n")

comb=(comb_simple+comb_complex)/2
print("Templates Combined----","\n")
print(comb,"\n")

predictions_combined = list(np.argmax(comb, axis=1))

print("Predictions","\n")
count_0 = predictions_combined.count(0)
count_1 = predictions_combined.count(1)
print('Comb  Not:', count_0)
print('Comb  Related:', count_1)

#np.savetxt('unlabeled_logits_combined_simple.txt',comb)
#print("")

#Se calcula el softmax 
t = 2
c=0
out = torch.softmax(torch.tensor(comb)/t, dim=1)
print("Probabilities","\n")
print(out)
np.savetxt('unlabeled_probabilities_combined.txt',out)


#Create data for destillation Classifier
#Se carga las entailment probabilities (probabilities) y las sentences (data)
with open("Data_for_Destilation_Classifier.json",'w',encoding="utf-8") as file:
    probabilities= open('/gaueko0/users/dromero012/Fine-Tuning_Tests/Version_1/unlabeled_probabilities_combined.txt','r').readlines()
    data= open("/gaueko0/users/dromero012/Fine-Tuning_Tests/Version_1/Data/V1_RAFT/Twitter_Complaints/twitter_complaints_data_test.txt",'r').readlines()

    for line in range(len(data)):
        c+=1
        content={"sentence":data[line].split('\n')[0],"label":out[line].tolist()}
        file.write(json.dumps(content)) # use `json.loads` to do the reverse

print(c)
