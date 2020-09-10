a=[1,1,2,3,3]

b=[]
print(b)
#b+=(a[:]==1) *3
b+=[True if i ==1 else False for i in a] *3

print(b)




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