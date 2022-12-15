import os
import requests

import warnings
warnings.filterwarnings('ignore')


def download_one_by_requests_basic_simple_auth(url, out, user, passw):
    """Download one file by requests lib with simple authorization.

    Args:
        url (str): The download url.
        out (str): The target file.
        user, passw (str): The user name and password for authorization.
    """

    r = requests.get(url, auth=(user, passw), verify=False)
    if r.status_code == 200:
        valid_out_file(out)
        with open(out, 'wb') as fd:
            fd.write(r.content)


def valid_out_file(out):
    """验证下载输出文件: 1.如果是文件目录，报错退出; 2.如果文件目录不存在则创建.

    Args:
        out (str): The input file.

    Raises:
        ValueError: If input file is a directory.
    """

    if os.path.isdir(out):
        raise ValueError('Outfile is a directory!')

    out_dir = os.path.dirname(out)
    if out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

if __name__ == '__main__':
    
    root_uri = 'https://scihub.copernicus.eu/dhus/odata/v1/'
    auth_name, auth_pw = ('mellem', '302112aa')

    output_dir = 'e:/S2'
    
    start_dt = '2022-10-19T00:00:00.000'
    end_dt = '2022-10-21T00:00:00.000'
    level = 'L2A'
    tile = '50SLH'
    # 50TLK 50TMK
    # 50SLJ 50SMJ
    # 50SLH 50SMH

    con = (f"Products?$format=json&"
           "$filter=year(IngestionDate) eq 2021 and "
           "month(IngestionDate) eq 12 and "
           "startswith(Name,'S2') and "
           "substringof('50SMH',Name) and "
           "substringof('L2A',Name)&"
           "$orderby=IngestionDate desc")
    con = ("Products?$format=json&"
           "$filter=IngestionDate gt datetime'{0}' and "
           "IngestionDate lt datetime'{1}' and "
           "startswith(Name,'S2') and "
           "substringof('{2}',Name) and "
           "substringof('{3}',Name)&"
           "$orderby=IngestionDate desc").format(start_dt, end_dt, tile, level)
    print(f"Query url: {root_uri + con}")
    r = requests.get(root_uri + con,
                     auth=(auth_name, auth_pw),
                     verify=False)
    if r.status_code == 200:
        content = r.json()
        results = content['d']['results']
        print(f'{len(results)} found!')
        FLAG = 1
        for result in results:
            product_id = result['Id']
            name = result['Name']
            product_url = root_uri + f"Products('{product_id}')/$value"
            preview_url = root_uri + f"Products('{product_id}')/Products('Quicklook')/$value"
            print(product_url + '_' + str(FLAG))
            r1 = requests.get(preview_url, auth=(auth_name, auth_pw), verify=False)
            download_one_by_requests_basic_simple_auth(preview_url, os.path.join(output_dir, f'{name}_{FLAG}.jpg'), auth_name, auth_pw)
            FLAG += 1
    
            wget_cmd = f"wget --no-check-certificate --content-disposition --continue --user={auth_name} --password={auth_pw} \"{product_url}\""
            # md5 = root_uri + f"Products('{product_id}')/Checksum/Value/$value"
            # m = requests.get(md5, auth=(auth_name, auth_pw), verify=False)
            # if m.status_code == 200:
            #     md5_str = m.content.decode()