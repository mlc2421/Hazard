#Hazard變數計算
data=pd.read_pickle("D:/研究所/碩二上/期末報告/statement 1960_2005.pkl")
bankruptcy=pd.read_pickle("D:/研究所/碩二上/期末報告/bankurptcy 1960_2006.pkl")
linktable=pd.read_pickle("D:/研究所/論文/感謝子建/link with date.pkl")


def DateVisual(target):

    if np.isnan(target)==False:
        a = pd.to_datetime("1960-01-01")
        b=(a+timedelta(days=int(target)))

        return b

#沒有破產的公司移除，並建立一個list，供往後使用
bankruptcy=bankruptcy.dropna(subset=["DLSTDT"])
bankruptcy=bankruptcy.reset_index(drop=True)
bankruptcy_list=bankruptcy["PERMNO"].unique().tolist()

#將Funda中日期可視化，並新增欄位用以裝載Permno和判斷是否為破產公司
data["Date"]=data["DATADATE"].apply(Paperfunction.DateVisual)
data["Permno"]=np.nan
data["bankruptcy"]=0



#將破產公司的破產日期可視化
bankruptcy["Bankruptcy Date"]=bankruptcy["DLSTDT"].apply(DateVisual)

#迴圈將財報檔案的資料加上Permno
for company in range(len(data)):

    print(company)

    #在linktable中，取出符合data的gvkey
    gvkey=data.iloc[company,0]
    linktable_filter=linktable[linktable["gvkey"] == gvkey]

    if linktable_filter.empty == False:

        linktable_filter=linktable_filter.reset_index(drop=True)

        for i in range(len(linktable_filter)):

            #若介於起始區間內，則存在permno
            if linktable_filter.iloc[i,9] <= data.iloc[company,13] <= linktable_filter.iloc[i,10] :

                data.iloc[company,14]=linktable_filter.iloc[i,4]

print("---"*50)

#將破產公司改為1

#第一層迴圈是用破產公司的permno下去跑
for permno in range(len(bankruptcy_list)):
    
    bankruptcy_permno=bankruptcy_list[permno]
    print(permno)

    # 先從Funda中取出permno列中有破產公司的部分，同時需要比對破產時間，故一併取出破產公司的資料列    
    data_filter=data[data["Permno"] == bankruptcy_permno]
    bankruptcy_filter=bankruptcy[bankruptcy["PERMNO"] == bankruptcy_permno]

    #若兩個篩選過後的資料皆有資料，即透過破產時間比對，並進行gvkey取得
    if data_filter.empty == False and bankruptcy_filter.empty == False:

        data_filter=data_filter.reset_index(drop=True)
        bankruptcy_filter=bankruptcy_filter.reset_index(drop=True)        

        #先找出破產時間，並將早於破產時間的部分資料撈出，設第一筆為正確gvkey
        bankruptcy_time=bankruptcy_filter.iloc[0,16]
        data_filter_before_bankruptcy=data_filter[data_filter["Date"] <= bankruptcy_time]
        bankruptcy_gvkey=data_filter_before_bankruptcy.iloc[0,0]

        #回到原先資料改以gvkey進行搜索
        data_filter_gvkey=data[data["GVKEY"] == bankruptcy_gvkey]
        data_filter_gvkey_time=data_filter_gvkey[data_filter_gvkey["Date"] >= bankruptcy_time]

        gvkey_bankruptcy_index=data_filter_gvkey_time.index.tolist()
        data.iloc[gvkey_bankruptcy_index,15] = 1


#控制變數新增
data["Log Asset"]=np.log(data["AT"])
data["ROA"]=data["NI"]/data["AT"]
data["Leverage"]=data["LT"]/data["AT"]
print(data)

        # # data.to_excel("C:/Users/user/Desktop/result.xlsx",encoding="cp950",index=None)
        # data.to_pickle("C:/Users/user/Desktop/剩下殘差標準差還沒好.pkl")


# 進行Beta跟Alpha的估計
stock=pd.read_pickle("D:/研究所/論文/感謝子建/Final all beta 2010_2019.pkl")
#將資料年份縮小至1960-2011間
stock=stock[(stock["year"] <=2011) & (stock["year"] >=1960)]
stock=stock.reset_index(drop=True)

#新創兩個column來裝alpha跟beta
stock["Alpha_12"]=np.nan
stock["Beta_12"]=np.nan

#新創兩個column來裝預期報酬跟殘差標準差
stock["Expected Return"]=np.nan
stock["STD_residual "]=np.nan

#以迴圈進行beta估計
for i in range(len(stock)):

    #區間必須是同一間公司才進行計算
    if stock.iloc[i-12,1] == stock.iloc[i,1]:
        print(i)
        
        #市場報酬加回無風險利率，並以市場模型進行估計
        individual=stock.iloc[i-12:i,20].values
        sp500=stock.iloc[i-12:i,25].values + stock.iloc[i-12:i,28].values
        sp500=sm.add_constant(sp500)

        z=sm.OLS(individual,sp500).fit()
        stock.iloc[i,35]=z.params[0]
        stock.iloc[i,36]=z.params[1]

        #以市場報酬佐個股系統風險計算出個股的預期報酬率
        realized=stock.iloc[i,20]
        expected=((stock.iloc[i,25] + stock.iloc[i,28])*stock.iloc[i,36])+stock.iloc[i,35]

        #殘差估計
        residual=realized-expected
        stock.iloc[i,37]=residual

        #前10個月的殘差標準差估計

        std_residual=np.std(stock.iloc[i-9:i+1,37])
        stock.iloc[i,38]=std_residual
