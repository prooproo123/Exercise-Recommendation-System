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
-pokusat ce se pretvaranje batcha sa .cpu()- rjeseno (problems.cpu() )

-greska : Dimension out of range (expected to be in range of [-1, 0], but got 1)
-prije toga je bilo obavljeno   helper = np.array(problems.cpu()).reshape(-1)

-ako se stavi -1 umjesto 1 greska, cini se greska u petlji dodavanje u target_index tako da se iterira 
-zakomentirana je druga petlja i dodavan je sa offsetom, stavljen je samo da appenda jednu listu na drugu

-greska u student model kod x+= pe output with shape [50400, 200] doesn't match the broadcast shape [1, 50400, 200]
-mice se unsqueeze(0) i stavljen je size 0 umjesto size 1

-greska u nekom retku CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle), navodno embedding layer dobiva krive indexe
-pada na ovoj liniji    res = self.multi_attn(query=self.layernorm(problems), key=x, value=x,
                              key_masks=None, query_masks=None, future_masks=None) --> DEAD END vjerojatno je neki out of range
                              
-kako pronaci rjesenje i da li odsutati od sakta?
'''