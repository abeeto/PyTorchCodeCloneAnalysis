import pyodbc
import pandas as pd


class DatabaseConnection:
    def __init__(self):
        """
        uid = 'j-gb-client'
        pwd = 'Mb1ygZywkGxi8ZQLbVeb'

        server = 's-kv-center-s27.officekiev.fozzy.lan'
        database = 'SalesHub.Dev'
        driver = '{ODBC Driver 17 for SQL Server}'
        conn = pyodbc.connect(driver=driver, server=server, database=database, uid=uid, pwd=pwd)

        self.connection = conn
        """

        self.connection = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                              "Server=S-KV-CENTER-S27;"
                              "Database=4t.Dev;"
                              "Trusted_Connection=yes;")


    def get_all_sales_by_articule_and_filial(self, articule, filial, start_date, end_date):
        sql_query_sales_by_articule_and_filial = f"select [Date], \
                QtySales as quantity \
                from [SalesHub.Dev].[DataHub].[v_SalesStores] \
                where LagerId = {articule}\
                and FilialId = {filial}\
                and [Date] >= {start_date} and [Date] <= {end_date}"
        df = pd.read_sql_query(sql_query_sales_by_articule_and_filial, self.connection)
        df = df.sort_values(by='Date')
        df.fillna({'quantity': 0}, inplace=True)
        return df

    def get_all_sales_by_articule_and_filial_with_prices(self, articule, filial, start_date, end_date):
        sql_query_sales_by_articule_and_filial = f"select [Date], \
                QtySales as quantity, \
                PriceOut as price \
                from [SalesHub.Dev].[DataHub].[v_SalesStores] \
                where LagerId = {articule}\
                and FilialId = {filial}\
                and [Date] >= {start_date} and [Date] <= {end_date}"
        df = pd.read_sql_query(sql_query_sales_by_articule_and_filial, self.connection)
        df = df.sort_values(by='Date')
        df.fillna({'quantity': 0}, inplace=True)
        df.fillna({'price': 0}, inplace=True)
        return df

    def get_all_sales_by_articule_and_filial_with_residues(self, articule, filial, start_date, end_date):
        sql_query_sales_by_articule_and_filial = f"select [Date], \
                QtySales as quantity, \
                StoreQtyDefault as residue, \
                PriceOut as price, \
                case when [ActivityId] is not null then 1 else 0 end as is_promo_day \
                from [SalesHub.Dev].[DataHub].[v_SalesStores] \
                where LagerId = {articule}\
                and FilialId = {filial}\
                and [Date] >= {start_date} and [Date] <= {end_date}"
        df = pd.read_sql_query(sql_query_sales_by_articule_and_filial, self.connection)
        df = df.sort_values(by='Date')
        df.fillna({'quantity': 0}, inplace=True)
        df.fillna({'residue': 0}, inplace=True)
        df.fillna({'price': 0}, inplace=True)
        return df

    def get_all_filial_articule_pairs(self):
        sql_query_filials_articules = f"select distinct FilialId as filial, LagerId as articule \
                from [SalesHub.Dev].[DataHub].[v_SalesStores] \
                order by FilialId, LagerId"
        df = pd.read_sql_query(sql_query_filials_articules, self.connection)
        return df

    def get_articules_for_filial(self, filial, start_date, end_date, min_days=0):
        sql_query_articules_by_filial = f"select LagerId \
                from [SalesHub.Dev].[DataHub].[v_SalesStores] \
                where FilialId = {filial} and [Date]>={start_date} and [Date]<={end_date} and QtySales > 0\
                group by LagerId\
                having count(distinct [Date]) >= {min_days}\
                order by LagerId"
        df = pd.read_sql_query(sql_query_articules_by_filial, self.connection)
        return df['LagerId'].tolist()

    def get_filials_for_articule(self, articule, start_date, end_date, min_days=0):
        sql_query_select_filials_for_articule = f"select FilialId \
                from [SalesHub.Dev].[DataHub].[v_SalesStores] \
                where LagerId={articule} and [Date] >= {start_date} and [Date] <= {end_date} and QtySales > 0\
                group by FilialId  \
                having count(distinct [Date]) >= {min_days}"
        df = pd.read_sql_query(sql_query_select_filials_for_articule, self.connection)
        return set(sorted(df['FilialId'].tolist()))

    def get_silpo_fora_trash_filials_set(self):
        sql_query_all_filials = \
            "select [filialId]\
            from [MasterData].[fil].[Filials]\
            where (businessId = 1 or businessId = 4 or businessId = 8)\
            and monitoringProgramId != 8\
            and  (filialName like 'Д %'  or filialName like 'С %' or filialName like 'ТР %')"
        df = pd.read_sql_query(sql_query_all_filials, self.connection)
        return set(df['filialId'].tolist())