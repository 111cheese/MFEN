import os
import mainin
# import chaofenbian_in


for i in range(4,6):
    os.system('python mainin.py')
    os.rename('E:/Dataset/Attention_ablash/ingf15_1.mat', 'E:/Dataset/Attention_ablash/ingf15_'+str(i)+'.mat')

# for j in range(1, 6):
#     os.system('python chaofenbian_in.py')