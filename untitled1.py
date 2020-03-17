# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:12:03 2019

@author: HPH
"""

from odps import ODPS
from odps.df import DataFrame

o = ODPS(
        access_id = 'LTAI4FozxnCup2e4sc4vhhVw',
        secret_access_key = 'hjHqxbxUpNczg6Pnbh0lt7weWtcJ0S',
        project = 'bi_pro',
        )

table = o.get_table('temp_hph_trd_sap_sales', project='bi_data_ana')
sale = DataFrame(table)

t = o.get_table('dwd_trd_sale_sap_i_d')
count = 0
for partition in t.partitions:
    print(partition.size)
    count += partition.size


c = [partition.size for partition in t.partitions]