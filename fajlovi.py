import pickle

#zadaci kandidati za preporuku
with open('arms.pkl', 'rb') as f:
    kandidati_zadaci = pickle.load(f)
#dict poveznica koncepti-zadaci
with open('q2kg.pkl', 'rb') as f:
    dict_koncepti_zadaci = pickle.load(f)
#utrenirani parametri DKVMN-CA modela
with open('shulun_param.pkl', 'rb') as f:
    parametri_kt_modela = pickle.load(f)

a=0

# """
# Evaluate the policy when it recommend exercises to different student
# allshulun:[[(923, 1), (175, 0), (1010, 1), (857, 0), (447, 0)], [........], [.........]]
# :param agent:
# :return: different students'predicted knowledge status
# """
# with open('./好未来数据/allshulun.pkl', 'rb') as f: