import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import spectral as spy
from sklearn import metrics
import time
from sklearn import preprocessing
import torch
import AGF_SLIC
import MyGCN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
samples_type = ['ratio', 'same_num'][0]
save_flag = False
use_gaussian_noise = True


def main(train_ratio, noise_std=0.01):
    

    for (FLAG, curr_train_ratio,Scale) in [(1,0.03,60)]:
    # for (FLAG, curr_train_ratio,Scale) in [(2,0.01,300)]:
    # for (FLAG, curr_train_ratio,Scale) in [(3,0.01,400)]:
    # for (FLAG, curr_train_ratio,Scale) in [(4,0.01,400)]:
        curr_train_ratio = train_ratio
        torch.cuda.empty_cache()
        OA_ALL = []
        AA_ALL = []
        KPP_ALL = []
        AVG_ALL = []
        Train_Time_ALL=[]
        Test_Time_ALL=[]
        
        Seed_List=[1]
        # Seed_List=[0,1]#随机种子点
        
        if FLAG == 1:
            data_mat = sio.loadmat('C:/Users/hhh/Desktop/JSJ/第四章程序/datasets/IP/indian_pines_corrected.mat')
            data = data_mat['indian_pines_corrected']
            gt_mat = sio.loadmat('C:/Users/hhh/Desktop/JSJ/第四章程序/datasets/IP/indian_pines_gt.mat')
            gt = gt_mat['indian_pines_gt']
            data_gf_mat = sio.loadmat('C:/Users/hhh/Desktop/JSJ/第四章数据/ingf30_4.mat')
            data_gf = data_gf_mat['ingf30_4']
            # 参数预设
            val_ratio = 0.03  # 验证集比例，与测试集比例一致
            class_count = 16  # 样本类别数
            learning_rate = 5e-4  # 学习率
            max_epoch = 350  # 迭代次数
            dataset_name = "indian_"  # 数据集名称
            # superpixel_scale=100
            pass
        if FLAG == 2:
            data_mat = sio.loadmat('D:/创新点2实验/dataset/PaviaU.mat')
            data = data_mat['paviaU']
            gt_mat = sio.loadmat('D:/创新点2实验/dataset/PaviaU_gt.mat')
            gt = gt_mat['paviaU_gt']
            data_gf_mat = sio.loadmat('./datasets/upgf30_5.mat')
            data_gf = data_gf_mat['upgf30_5']
            # 参数预设
            val_ratio = 0.01  # 验证集比例，与测试集比例一致
            class_count = 9  # 样本类别数
            learning_rate = 5e-4  # 学习率
            max_epoch = 400  # 迭代次数
            dataset_name = "paviaU_"  # 数据集名称
            # superpixel_scale = 100
            pass
        if FLAG == 3:
            data_mat = sio.loadmat('D:/创新点2实验/dataset/Salinas_corrected.mat')
            data = data_mat['salinas_corrected']
            gt_mat = sio.loadmat('D:/创新点2实验/dataset/Salinas_gt.mat')
            gt = gt_mat['salinas_gt']
            data_gf_mat = sio.loadmat('./datasets/sagf30_5.mat')
            data_gf = data_gf_mat['sagf30_5']
            # 参数预设
            val_ratio = 0.005  # 验证集比例，与测试集比例一致
            class_count = 16  # 样本类别数
            learning_rate = 5e-4  # 学习率
            max_epoch = 300  # 迭代次数
            dataset_name = "salinas_"  # 数据集名称
            # superpixel_scale = 100
            pass
        if FLAG == 4:
            data_mat = sio.loadmat('D:/创新点2实验/dataset/WHU_Hi_LongKou.mat')
            data = data_mat['WHU_Hi_LongKou']
            gt_mat = sio.loadmat('D:/创新点2实验/dataset/WHU_Hi_LongKou_gt.mat')
            gt = gt_mat['WHU_Hi_LongKou_gt']
            data_gf_mat = sio.loadmat('./datasets/WHU_LK_gf1.mat')
            data_gf = data_gf_mat['gf_1']
            
            # 参数预设
            val_ratio = 0.01  # 验证集比例，与测试集比例一致
            class_count = 9  # 样本类别数
            learning_rate = 5e-4  # 学习率
            max_epoch = 300  # 迭代次数
            dataset_name = "WHU_"  # 数据集名称
            # superpixel_scale = 200
            pass
        ###########
        superpixel_scale=Scale#########################
        train_samples_per_class = curr_train_ratio  # 当定义为每类样本个数时,则该参数更改为训练样本数
        val_samples = class_count
        train_ratio = curr_train_ratio
        # cmap = cm.get_cmap('jet', class_count + 1)
        # plt.set_cmap(cmap)
        m, n, d = data.shape  # 高光谱数据的三个维度

        # 在标准化前可选地添加高斯噪声，便于做鲁棒性实验
        if use_gaussian_noise:
            data = data.astype(np.float32, copy=False)
            noise = np.random.normal(0.0, noise_std, size=data.shape).astype(np.float32)
            data = data + noise

        # 数据standardization标准化,即提前全局BN
        orig_data=data
        height, width, bands = data.shape  # 原始高光谱数据的三个维度
        data = np.reshape(data, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        data = np.reshape(data, [height, width, bands])
        
        # # 打印每类样本个数
        # gt_reshape=np.reshape(gt, [-1])
        # for i in range(class_count):
        #     idx = np.where(gt_reshape == i + 1)[-1]
        #     samplesCount = len(idx)
        #     print(samplesCount)
        
        def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
            '''
            get classification map , then save to given path
            :param label: classification label, 2D
            :param name: saving path and file's name
            :param scale: scale of image. If equals to 1, then saving-size is just the label-size
            :param dpi: default is OK
            :return: null
            '''
            fig, ax = plt.subplots()
            numlabel = np.array(label)
            v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
            ax.set_axis_off()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
            foo_fig = plt.gcf()  # 'get current figure'
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
            pass
        def draw_map(label, background):
            plot_max = np.array(label)+1
            h = plot_max.shape[0]
            w = plot_max.shape[1]
            plot_max = np.reshape(plot_max, [-1])
            plot_max[background] = 0
            plot_max = np.reshape(plot_max, (h,w))
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
            plt.axis('off')
            plt.figure(figsize=(w / 10, h / 10), dpi=100)  # set size
            plt.xlim(0, w)
            plt.ylim(h, 0)
            plt.pcolor(plot_max, cmap='jet')
            plt.savefig('./results/SS1%' + '.png', format='png')
            plt.show()
            plt.close()
        
        def GT_To_One_Hot(gt, class_count):
            '''
            Convet Gt to one-hot labels
            :param gt:
            :param class_count:
            :return:
            '''
            GT_One_Hot = []  # 转化为one-hot形式的标签
            for i in range(gt.shape[0]):
                for j in range(gt.shape[1]):
                    temp = np.zeros(class_count,dtype=np.float32)
                    if gt[i, j] != 0:
                        temp[int( gt[i, j]) - 1] = 1
                    GT_One_Hot.append(temp)
            GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
            return GT_One_Hot
    
        for curr_seed in Seed_List:
            # step2:随机10%数据作为训练样本。方式：给出训练数据与测试数据的GT
            random.seed(curr_seed)
            gt_reshape = np.reshape(gt, [-1])
            train_rand_idx = []
            val_rand_idx = []
            if samples_type == 'ratio':
                for i in range(class_count):
                    idx = np.where(gt_reshape == i + 1)[-1]
                    samplesCount = len(idx)
                    rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                    rand_idx = random.sample(rand_list,
                                            np.ceil(samplesCount * train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
                    rand_real_idx_per_class = idx[rand_idx]
                    train_rand_idx.append(rand_real_idx_per_class)
                train_rand_idx = np.array(train_rand_idx, dtype=object)
                train_data_index = []
                for c in range(train_rand_idx.shape[0]):
                    a = train_rand_idx[c]
                    for j in range(a.shape[0]):
                        train_data_index.append(a[j])
                train_data_index = np.array(train_data_index)  # set [314] 训练集像素索引
                
                ##将测试集（所有样本，包括训练样本）也转化为特定形式
                train_data_index = set(train_data_index)
                all_data_index = [i for i in range(len(gt_reshape))]
                all_data_index = set(all_data_index)  # set 所有像素索引
                
                # 背景像元的标签
                background_idx = np.where(gt_reshape == 0)[-1]
                background_idx = set(background_idx)
                test_data_index = all_data_index - train_data_index - background_idx
                
                # 从测试集中随机选取部分样本作为验证集
                val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))  # 验证集数量
                test_data_seq = sorted(test_data_index)  # set -> sequence
                val_data_count = min(val_data_count, len(test_data_seq))  # 防止越界
                val_data_index = random.sample(test_data_seq, val_data_count)
                val_data_index = set(val_data_index)
                test_data_index = test_data_index - val_data_index  # 由于验证集为从测试集分裂出，所以测试集应减去验证集
                
                # 将训练集 验证集 测试集 整理 都是索引
                test_data_index = list(test_data_index)
                train_data_index = list(train_data_index)
                val_data_index = list(val_data_index)
            
            if samples_type == 'same_num':
                for i in range(class_count):
                    idx = np.where(gt_reshape == i + 1)[-1]
                    samplesCount = len(idx)
                    real_train_samples_per_class = train_samples_per_class
                    rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                    if real_train_samples_per_class > samplesCount:
                        real_train_samples_per_class = samplesCount
                    rand_idx = random.sample(rand_list,
                                            real_train_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
                    rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
                    train_rand_idx.append(rand_real_idx_per_class_train)
                train_rand_idx = np.array(train_rand_idx, dtype=object)
                val_rand_idx = np.array(val_rand_idx)
                train_data_index = []
                for c in range(train_rand_idx.shape[0]):
                    a = train_rand_idx[c]
                    for j in range(a.shape[0]):
                        train_data_index.append(a[j])
                train_data_index = np.array(train_data_index)
                
                train_data_index = set(train_data_index)
                all_data_index = [i for i in range(len(gt_reshape))]
                all_data_index = set(all_data_index)
                
                # 背景像元的标签
                background_idx = np.where(gt_reshape == 0)[-1]
                background_idx = set(background_idx)
                test_data_index = all_data_index - train_data_index - background_idx
                
                # 从测试集中随机选取部分样本作为验证集
                val_data_count = int(val_samples)  # 验证集数量
                test_data_seq = sorted(test_data_index)  # set -> sequence
                val_data_count = min(val_data_count, len(test_data_seq))  # 防止越界
                val_data_index = random.sample(test_data_seq, val_data_count)
                val_data_index = set(val_data_index)
                
                test_data_index = test_data_index - val_data_index
                # 将训练集 验证集 测试集 整理
                test_data_index = list(test_data_index)
                train_data_index = list(train_data_index)
                val_data_index = list(val_data_index)
            
            # 获取训练样本的标签图
            train_samples_gt = np.zeros(gt_reshape.shape)
            for i in range(len(train_data_index)):
                train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
                pass
            
            # 获取测试样本的标签图
            test_samples_gt = np.zeros(gt_reshape.shape)
            for i in range(len(test_data_index)):
                test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
                pass
            
            Test_GT = np.reshape(test_samples_gt, [m, n])  # 测试样本图
            
            # 获取验证集样本的标签图
            val_samples_gt = np.zeros(gt_reshape.shape)
            for i in range(len(val_data_index)):
                val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
                pass

            train_samples_gt=np.reshape(train_samples_gt,[height,width])
            test_samples_gt=np.reshape(test_samples_gt,[height,width])
            val_samples_gt=np.reshape(val_samples_gt,[height,width])

            train_samples_gt_onehot=GT_To_One_Hot(train_samples_gt,class_count)
            test_samples_gt_onehot=GT_To_One_Hot(test_samples_gt,class_count)
            val_samples_gt_onehot=GT_To_One_Hot(val_samples_gt,class_count)

            train_samples_gt_onehot=np.reshape(train_samples_gt_onehot,[-1,class_count]).astype(int)
            test_samples_gt_onehot=np.reshape(test_samples_gt_onehot,[-1,class_count]).astype(int)
            val_samples_gt_onehot=np.reshape(val_samples_gt_onehot,[-1,class_count]).astype(int)

            ############制作训练数据和测试数据的gt掩膜.根据GT将带有标签的像元设置为全1向量##############
            # 训练集
            train_label_mask = np.zeros([m * n, class_count])
            temp_ones = np.ones([class_count])
            train_samples_gt = np.reshape(train_samples_gt, [m * n])
            for i in range(m * n):
                if train_samples_gt[i] != 0:
                    train_label_mask[i] = temp_ones
            train_label_mask = np.reshape(train_label_mask, [m* n, class_count])

            # 测试集
            test_label_mask = np.zeros([m * n, class_count])
            temp_ones = np.ones([class_count])
            test_samples_gt = np.reshape(test_samples_gt, [m * n])
            for i in range(m * n):
                if test_samples_gt[i] != 0:
                    test_label_mask[i] = temp_ones
            test_label_mask = np.reshape(test_label_mask, [m* n, class_count])

            # 验证集
            val_label_mask = np.zeros([m * n, class_count])
            temp_ones = np.ones([class_count])
            val_samples_gt = np.reshape(val_samples_gt, [m * n])
            for i in range(m * n):
                if val_samples_gt[i] != 0:
                    val_label_mask[i] = temp_ones
            val_label_mask = np.reshape(val_label_mask, [m* n, class_count])
            
            
            ls = AGF_SLIC.AGF_SLIC(data_gf, np.reshape( train_samples_gt,[height,width]), class_count-1)
            tic0=time.time()
            Q, S ,A,Seg= ls.simple_superpixel(scale=superpixel_scale)
            toc0 = time.time()
            AGF_SLIC_Time=toc0-tic0
            # np.save(dataset_name+'Seg',Seg)
            print("AGF-SLIC costs time: {}".format(AGF_SLIC_Time))
            Q=torch.from_numpy(Q).to(device)
            A=torch.from_numpy(A).to(device)

            #转到GPU
            train_samples_gt=torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
            test_samples_gt=torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
            val_samples_gt=torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)
            #转到GPU
            train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
            test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
            val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)
            #转到GPU
            train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
            test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
            val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)
            
            #____________________________________________________________________
            #model
            net_input=np.array( data,np.float32)
            net_input=torch.from_numpy(net_input.astype(np.float32)).to(device)

            if dataset_name == "indian_":
                net = MyGCN.MyGCN(height, width, bands, class_count, Q, A, model='smoothed')
            else:
                net = MyGCN.MyGCN(height, width, bands, class_count, Q, A)

            print("parameters", net.parameters(), len(list(net.parameters())))
            net.to(device)
            total_params = 0
            for param in net.parameters():
                # numel() 方法返回张量中元素的总数
                total_params += param.numel()
            print(f"模型的总参数量: {total_params}")
            trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print(f"训练参数量: {trainable_params}")
            def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
                real_labels = reallabel_onehot
                we = -torch.mul(real_labels,torch.log(predict))
                we = torch.mul(we, reallabel_mask)
                pool_cross_entropy = torch.sum(we)
                return pool_cross_entropy
            

            zeros = torch.zeros([m * n]).to(device).float()
            def evaluate_performance(network_output,train_samples_gt,train_samples_gt_onehot, require_AA_KPP=False,printFlag=True):
                if False==require_AA_KPP:
                    with torch.no_grad():
                        available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
                        available_label_count=available_label_idx.sum()#有效标签的个数
                        correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
                        OA= correct_prediction.cpu()/available_label_count
                        
                        return OA
                else:
                    with torch.no_grad():
                        #计算OA
                        available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
                        available_label_count=available_label_idx.sum()#有效标签的个数
                        correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
                        OA= correct_prediction.cpu()/available_label_count
                        OA=OA.cpu().numpy()
                        
                        # 计算AA
                        zero_vector = np.zeros([class_count])
                        output_data=network_output.cpu().numpy()
                        train_samples_gt=train_samples_gt.cpu().numpy()
                        train_samples_gt_onehot=train_samples_gt_onehot.cpu().numpy()
                        
                        output_data = np.reshape(output_data, [m * n, class_count])
                        idx = np.argmax(output_data, axis=-1)
                        for z in range(output_data.shape[0]):
                            if ~(zero_vector == output_data[z]).all():
                                idx[z] += 1
                        # idx = idx + train_samples_gt
                        count_perclass = np.zeros([class_count])
                        correct_perclass = np.zeros([class_count])
                        for x in range(len(train_samples_gt)):
                            if train_samples_gt[x] != 0:
                                count_perclass[int(train_samples_gt[x] - 1)] += 1
                                if train_samples_gt[x] == idx[x]:
                                    correct_perclass[int(train_samples_gt[x] - 1)] += 1
                        test_AC_list = correct_perclass / count_perclass
                        test_AA = np.average(test_AC_list)

                        # 计算KPP
                        test_pre_label_list = []
                        test_real_label_list = []
                        output_data = np.reshape(output_data, [m * n, class_count])
                        idx = np.argmax(output_data, axis=-1)
                        idx = np.reshape(idx, [m, n])
                        for ii in range(m):
                            for jj in range(n):
                                if Test_GT[ii][jj] != 0:
                                    test_pre_label_list.append(idx[ii][jj] + 1)
                                    test_real_label_list.append(Test_GT[ii][jj])
                        test_pre_label_list = np.array(test_pre_label_list)
                        test_real_label_list = np.array(test_real_label_list)
                        kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                        test_real_label_list.astype(np.int16))
                        test_kpp = kappa

                        # 输出
                        if printFlag:
                            print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                            print('acc per class:')
                            print(test_AC_list)

                        OA_ALL.append(OA)
                        AA_ALL.append(test_AA)
                        KPP_ALL.append(test_kpp)
                        AVG_ALL.append(test_AC_list)
                        if save_flag:
                            # 保存数据信息
                            f = open('results\\' + dataset_name + '_results.txt', 'a+')
                            str_results = '\n======================' \
                                        + " learning rate=" + str(learning_rate) \
                                        + " epochs=" + str(max_epoch) \
                                        + " train ratio=" + str(train_ratio) \
                                        + " val ratio=" + str(val_ratio) \
                                        + " ======================" \
                                        + "\nOA=" + str(OA) \
                                        + "\nAA=" + str(test_AA) \
                                        + '\nkpp=' + str(test_kpp) \
                                        + '\nacc per class:' + str(test_AC_list) + "\n"
                                        # + '\ntrain time:' + str(time_train_end - time_train_start) \
                                        # + '\ntest time:' + str(time_test_end - time_test_start) \
                            f.write(str_results)
                            f.close()
                            return OA
            
            # 训练
            optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)#,weight_decay=0.0001
            best_loss=99999
            net.train()
            tic1 = time.perf_counter()
            for i in range(max_epoch+1):
                optimizer.zero_grad()  # zero the gradient buffers
                output= net(net_input)
                loss = compute_loss(output,train_samples_gt_onehot,train_label_mask)
                loss.backward(retain_graph=False)
                optimizer.step()  # Does the update
                torch.cuda.empty_cache()
                if i%10==0:
                    with torch.no_grad():
                        net.eval()
                        output= net(net_input)
                        trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
                        trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
                        valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
                        valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)
                        print("{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA, valloss, valOA))

                        if valloss < best_loss :
                            best_loss = valloss
                            torch.save(net.state_dict(),"./model/best_model0115.pt")
                            print('save model...')
                    torch.cuda.empty_cache()
                    net.train()

            toc1 = time.perf_counter()
            print("\n\n====================training done. starting evaluation...========================\n")
            training_time=toc1 - tic1 + AGF_SLIC_Time #分割耗时需要算进去
            Train_Time_ALL.append(training_time)
            
            torch.cuda.empty_cache()
            with torch.no_grad():
                net.load_state_dict(torch.load("./model/best_model0115.pt"))
                net.eval()
                tic2 = time.perf_counter()
                output = net(net_input)
                toc2 = time.perf_counter()
                testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
                testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot,require_AA_KPP=True,printFlag=False)
                print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))
                #计算
                classification_map=torch.argmax(output, 1).reshape([height,width]).cpu()+1
                background_idx1 = np.where(gt_reshape == 0)[-1]

                # draw_map(classification_map,background_idx1)

                testing_time=toc2 - tic2
                Test_Time_ALL.append(testing_time)
                ## Saving data
                # sio.savemat(dataset_name+"softmax",{'softmax':output.reshape([height,width,-1]).cpu().numpy()})
                # np.save(dataset_name+"A_1", A_1.cpu().numpy())
                # np.save(dataset_name+"A_2", A_2.cpu().numpy())
                
            torch.cuda.empty_cache()

            del net
            
        OA_ALL = np.array(OA_ALL)
        AA_ALL = np.array(AA_ALL)
        KPP_ALL = np.array(KPP_ALL)
        AVG_ALL = np.array(AVG_ALL)
        Train_Time_ALL=np.array(Train_Time_ALL)
        Test_Time_ALL=np.array(Test_Time_ALL)


        print("\ntrain_ratio={}".format(curr_train_ratio),
            "\n==============================================================================")
        print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
        print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
        print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
        print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
        print("Average training time:{}".format(np.mean(Train_Time_ALL)))
        print("Average testing time:{}".format(np.mean(Test_Time_ALL)))
        
        return np.mean(OA_ALL), np.mean(AA_ALL), np.mean(KPP_ALL),np.mean(AVG_ALL, 0), np.mean(Train_Time_ALL), np.mean(Test_Time_ALL)
        if save_flag:
            # 保存数据信息
            f = open('./results/' + dataset_name + '_results0115.txt', 'a+')
            str_results = '\n\n************************************************' \
            +"\ntrain_ratio={}".format(curr_train_ratio) \
            +'\nOA='+ str(np.mean(OA_ALL))+ '+-'+ str(np.std(OA_ALL)) \
            +'\nAA='+ str(np.mean(AA_ALL))+ '+-'+ str(np.std(AA_ALL)) \
            +'\nKpp='+ str(np.mean(KPP_ALL))+ '+-'+ str(np.std(KPP_ALL)) \
            +'\nAVG='+ str(np.mean(AVG_ALL,0))+ '+-'+ str(np.std(AVG_ALL,0)) \
            +"\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
            +"\nAverage testing time:{}".format(np.mean(Test_Time_ALL))
            f.write(str_results)
            f.close()
        

    
if __name__ == '__main__':
    import os
    file_name = './output/MFEN/IP_gaussian_noise_std.txt'
    if not os.path.exists(file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    # for train_ratio in [0.001, 0.005, 0.01, 0.02]:
    # for train_ratio in [0.01, 0.02, 0.04, 0.05]:
    train_ratio = 0.03
    aussian_noise_std_list = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    for std_value in aussian_noise_std_list:
        results = []
        print(f"Training with train_ratio: {train_ratio}")
        with open(file_name, 'a') as f:
            f.write(f"Training with train_ratio: {train_ratio}\n")
            f.write(f"\n--- gaussian_noise_std: {std_value} ---\n")
        for i in range(5):
            oa, aa, kappa, ac_list, training_time, testing_time = main(train_ratio, noise_std=std_value, exp_seed=seed)
            results.append((oa, aa, kappa, ac_list, training_time, testing_time))
            with open(file_name, 'a') as f:
                f.write(f"gen: {i+1}, oa: {oa}, aa: {aa}, kappa: {kappa}\n")
                f.write(f"AA per class: {ac_list}\n")
        avg_oa = np.mean([result[0] for result in results])
        avg_aa = np.mean([result[1] for result in results])
        avg_kappa = np.mean([result[2] for result in results])
        avg_ac_list = np.mean([result[3] for result in results], axis=0)
        avg_train_time = np.mean([result[4] for result in results])
        avg_test_time = np.mean([result[5] for result in results])
        with open(file_name, 'a') as f:            
            f.write(f"Average results for gaussian_noise_std {std_value}:\n, oa: {avg_oa}, aa: {avg_aa}, kappa: {avg_kappa}\n")
            f.write(f"Average AA per class: {avg_ac_list}\n")
            f.write(f"Average training time: {avg_train_time:.2f}s\n")
            f.write(f"Average testing time: {avg_test_time:.2f}s\n")
            f.write("===============================================================\n")
    
    
    
    
    
