def get_dict_symbol_2_p_value(disease):
    df_dif_exp = pd.read_csv('../data/GEO/Dif_Exp_Res.tsv', sep= '\t')
    lst_dif_exp_symbol = df_dif_exp['Gene.Symbol'].tolist()
    lst_dif_exp_p_value = df_dif_exp['P.Value'].tolist()
    lst_dif_exp_probid = df_dif_exp['ID'].tolist()
    dict_symbol2pvalue = {}

    gse164760_genelist = []
    for x in lst_dif_exp_symbol:
        if x == '---':
            continue
        try:
            tmp_lst = x.split(" /// ")
        except:
            continue
        for y in tmp_lst:
            gse164760_genelist.append(y)

    gse164760_genelist = sorted(list(set(gse164760_genelist)))
    dict_genesymbol2probid = {}

    for gene in gse164760_genelist:
        dict_genesymbol2probid[gene] = []

    for i in range(len(lst_dif_exp_symbol)):
        tmp_x = lst_dif_exp_symbol[i]
        tmp_probid = lst_dif_exp_probid[i]
        if tmp_x == '---':
            continue
        try:
            tmp_lst = tmp_x.split(" /// ")
        except:
            continue
        for gene in tmp_lst:
            dict_genesymbol2probid[gene].append(tmp_probid)
            break

    dict_probid2pvalue = {}
    for i in range(len(lst_dif_exp_p_value)):
        tmp_probid = lst_dif_exp_probid[i]
        tmp_pvalue = lst_dif_exp_p_value[i]
        dict_probid2pvalue[tmp_probid] = tmp_pvalue
    for gene in gse164760_genelist:
        tmp_probid_lst = dict_genesymbol2probid[gene]
        tmp_pvalue = dict_probid2pvalue[tmp_probid_lst[0]]
        dict_symbol2pvalue[gene] = tmp_pvalue
    return dict_symbol2pvalue
    
    
def Reward_supervision_driven(z_scores, D_d, D_n):
    pos_reward = np.mean([z_scores[g] for g in D_d])
    neg_reward = np.mean([z_scores[g] for g in D_n])
    return pos_reward - neg_reward
    
def Reward_data_driven(z_scores):
    reward_data_driven = 0
    cnt = 0
    for x in nodelist:
        tmp_idx = dict_symbol2idx[x]
        tmp_zscore = z_scores[tmp_idx]
        if tmp_zscore < 2:
            continue
        cnt += 1
        tmp_pvalue = dict_symbol2pvalue_global[x]
        reward_data_driven += np.log10(tmp_pvalue+ 1e-10)
    reward_data_driven = reward_data_driven / (cnt+1)
    return reward_data_driven