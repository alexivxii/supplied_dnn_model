https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/ https://stackoverflow.com/questions/46308374/what-is-validation-data-used-for-in-a-keras-sequential-model

groceries_list - lista de produse pe care poate sa le adauge utilizatorul in lista de cumparaturi, cantitatea nu conteaza

dataset_size = numarul de persoane pentru care avem/generam date

weeks_per_person - numarul de saptamani pt fecare persoana, fiecare persoana adauga o singura lista de cumparaturi pe saptamana

noi prezicem a cincea saptamana bazata pe listele de cumparaturi din primele 4 saptamani

generam seturile de date

x = produsele adaugate (0/1) in listele de cumparaturi, generate cu random (cuprinde toate cele 4 liste concatenate)
y = lista de cumparaturi generata prin average pt a 5 a saptamana

impartim setul de date in train si test 4000 si 1000

setul de date de test e un set pe care modelul nu l-a vazut la antrenare absolut deloc

get_model - definim arhitectura modelului; 3 straturi de neuroni (deep neuronal network dens)

model = get_new_compiled_mode()
	pentru a compila un model de trebuie o arhitectura, o functie pierdere si un optimizer
	functia asta de pierdere face diferenta dintre ce a generat si ce trebuia sa genereze (bazat pe x_test, modelul genereaza un y si pe acesta il compara cu y_test, la train la fel)  MeanSquaredError = ce ne asteptam - ce am generat la patrat

gata, s a generat modelul

in fit_model -> AICI SE FACE EFECTIV ANTRENAREA (INVATAREA), asta e functie din keras

-----------------

De ce avem doua metrici?

modelul isi calculeaza ajustarea pe baza metricii mse, dar noi urmarim si mae (eroarea nepatratica din formula aia, suma din ce era in paranteza dar fara patrat)

El la antrenare isi separa datele de train in train si validation la fiecare etapa (asta face functia acolo in spate pe treaba ei)

validarea la fiecare etapa e ca sa urmarim ca la fiecare etapa invata ceva

------------------------------------------- s a terminat antrenarea

ACUM AJUNGEM LA TESTAREA MODELULUI PE SPLITUL DE DATE PE CARE NU LE-A VAZUT NICIODATA

loss, mae = model.evaluate(x_test, y_test)

aici din nou modelul, bazat pe ce a invatat, din x_test o sa genereze un y pe care apoi il compara cu y_test

erorile astea fiind foarte mici, inseamna ca face bine





