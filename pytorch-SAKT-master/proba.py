import main
import split_dataset


split_dataset.split('../data/biology30/biology30.csv',sep='\t')
#main.main('../data/skill_builder/skill_builder_data_sakt_train.csv','../data/skill_builder/skill_builder_data_sakt_test.csv',sep='\t')

main.main(r'C:\Z_Sucic\Exercise-Recommendation-System\data\biology30\biology30_sakt_train.csv',r'C:\Z_Sucic\Exercise-Recommendation-System\data\biology30\biology30_sakt_test.csv',sep=',')


'''
student_model- salje se velicina num_skills *2 +1   CINI SE KRIVO
treba pogledati u torch.nn kakav format embeddinga treba biti
dataset.py - zasto uzimanje do predzadnjeg i od drugog clana, zar ne bi trebali biti svi clanovi?
            -kakvog formata treba biti x, treba li true/false ili 1/0
            - kod __getitem__ zasto se nakon pretvorbe u true/false mnozi sa max_skill_num, koja je poanta toga?
-num_skill se uzima kao najveci index u listi pitanja- to treba svugdje izmijenti da ne bude zbunjujuce->ili napraviti da je len(questions) ili preimenovait u max_index

-treba isprobati i na biologiji i assistmentsu

KAD SE USPIJE POKRETATI
-isprobati razne optimizere

'''


'''
Debagiranje preko colaba

-kada se self.batch[k] pretvori u tensor pomocu self.batch[k]=torch.cat(self.batch[k] prolazi i dode do run_epoch
- daje gresku: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
-i dalje zeza print debug
-pokusat ce se pretvaranje batcha sa batch = batch.cpu().data.numpy()

'''