#將前面整理好的檔案與殘差標準差合併
data=pd.read_pickle("D:/研究所/碩二上/期末報告/完整財報資料.pkl")
stock=pd.read_pickle("D:/研究所/碩二上/期末報告/stock with std 1960_2011.pkl")
allfunda=pd.read_pickle("D:/研究所/論文/感謝子建/funda 2020.pkl")

funda=allfunda[(allfunda["indfmt"]=="INDL") & (allfunda["datafmt"]=="STD") & (allfunda["consol"]=="C") & (allfunda["popsrc"]=="D") & (allfunda["fyear"] <= 2005) & (allfunda["fyear"] >= 1960)]
funda=funda.reset_index(drop=True)

#先將處理好的變數資料與老闆的市值資料合併
data=pd.merge(data,funda[["gvkey","fyear","fyr","cusip","prcc_f","csho","ceq"]],how="left",left_on=["GVKEY","FYEAR","FYR"],right_on=["gvkey","fyear","fyr"])
data["M/B Ratio"]=data["prcc_f"]*data["csho"]/data["ceq"]

#我她媽column命名出多一個空格= =，另外將有破產公司後面年分，permno可能消失的，以之前的permno補上
stock.rename(columns={"STD_residual ":"STD_residual"}, inplace=True)
data["Permno"]=data["Permno"].fillna(data.groupby("GVKEY")["Permno"].transform("mean"))



#檔案Merge，並將檔案中無限大的部分改以Nan，最後將四個變數中有缺失值得部分皆移除
result=pd.merge(data,stock[["PERMNO","DATE","RETX","mktrf","rf","year",\
                "month","Alpha_12","Beta_12","Expected Return","STD_residual"]],how="left",
                left_on=["FYEAR","FYR","Permno"],right_on=["year","month","PERMNO"])

result=result.replace([np.inf, -np.inf], np.nan)
result=result.dropna(subset=["bankruptcy","Log Asset","ROA","Leverage","STD_residual"])



#要獲得各年度的Logistic估計結果，所以以年份作為迴圈對象，先取的要估計的1980-2005年份數字
year_list=data["FYEAR"].unique().tolist()
estimate_list=[i for i in year_list if 1980<= i <=2005 ]



#建立一個Dataframe，來裝Logistic的估計結果
logistic_df=pd.DataFrame({ "year" : estimate_list })
logistic_df["Constant"]=np.nan
logistic_df["Log Asset_L"]=np.nan
logistic_df["ROA_L"]=np.nan
logistic_df["Leverage_L"]=np.nan
logistic_df["STD_residual_L"]=np.nan
logistic_df["P-Value of Log Asset"]=np.nan
logistic_df["P-Value of ROA"]=np.nan
logistic_df["P-Value of Leverage"]=np.nan
logistic_df["P-Value of STD_residual"]=np.nan

#以迴圈開始進行估計
for year in range(len(estimate_list)):

    for i in range(len(logistic_df)):

        if logistic_df.iloc[i,0] == estimate_list[year]:

            #將Result中各年度的前17年到前1年先篩選出來
            result_filter=result[(result["FYEAR"] >= estimate_list[year]-17) & (result["FYEAR"] <= estimate_list[year]-1)]
            result_filter=result_filter.reset_index(drop=True)

            #若不是空白Dataframe就先取出解釋變數，及被解釋變數           
            if result_filter.empty == False:
                x=result_filter[["Log Asset","ROA","Leverage","STD_residual"]]
                x=sm.add_constant(x)
                y=result_filter[["bankruptcy"]]

                #只要該年分存在至少一間破產公司就開始進行Logistic估計
                if y["bankruptcy"].sum() >= 1:

                    beta=sm.Logit(y,x).fit()
                    logistic_df.iloc[i,1]=beta.params[0]
                    logistic_df.iloc[i,2]=beta.params[1]
                    logistic_df.iloc[i,3]=beta.params[2]
                    logistic_df.iloc[i,4]=beta.params[3]
                    logistic_df.iloc[i,5]=beta.params[4]
                    logistic_df.iloc[i,6]=beta.pvalues[1]
                    logistic_df.iloc[i,7]=beta.pvalues[2]
                    logistic_df.iloc[i,8]=beta.pvalues[3]
                    logistic_df.iloc[i,9]=beta.pvalues[4]


print(logistic_df)
print("---"*50)

# 將Logistic回歸結果與前面財報數據合併
result=pd.merge(result,logistic_df,how="left",left_on=["FYEAR"],right_on=["year"])

result["Score"]=result["Constant"]+result["Log Asset"]*result["Log Asset_L"]+result["ROA"]*result["ROA_L"]\
                        +result["Leverage"]*result["Leverage_L"]+result["STD_residual"]*result["STD_residual_L"]

result["Hazard Ratio"]=np.exp(result["Score"])/(1+np.exp(result["Score"]))

# #僅保留需要的數據，並存成Pickle檔
result=result[["GVKEY","Permno","cusip","FYEAR","FYR","AT","LT","Log Asset","M/B Ratio","Leverage","Hazard Ratio"]]
print(result)

result.to_pickle("C:/Users/user/Desktop/完整Hazard 0121.pkl")
