import openstack
from openstack.config import loader
from openstack import utils
import cinderclient as cinder 
import novaclient as nova
import glanceclient as glance
import swiftclient as swift

import time
import os
import sys
import subprocess

def clear_ip_cache(ip):
  os.system('ssh-keygen -f "/home/aligula/.ssh/known_hosts" -R '+ip)
  print ('cleared old ssh ip cache')


def read_identity():
  identity = open('identity.txt')
  identity_dict = {}
  while True:
    line = identity.readline()
    if len(line)==0:
      return identity_dict
    else:
      line = line.split(',')
      if len(line)==2:
          identity_dict[line[0].lstrip(' ')]=line[-1].lstrip(' ').replace('\n','')
  
def create_connection():
  return openstack.connect(
    auth=read_identity(),
    region_name = "RegionOne",
    interface =  "public",
    identity_api_version='3'
  )
#  the server has to wait for file transfer to start doing any computation

def create_server(connection):
  server = connection.create_server('testserver4',
                        image='d8a9f636-5776-4c8f-94f0-ab0a19113762',
                        #filename = './userdata',
                        flavor = 'm1.small',
                        key_name = 'ruiyanghp',
                        userdata = '#!/bin/sh\n' 
                              +'apt-get update \n ' +
                              'echo heyhye\n'+
                              'apt-get install -y python3-pip \n'+
                              'apt-get install -y git\n'+
                              'git clone https://github.com/dylanrchen/openstack_torchmodel /home/ubuntu/openstack_torchmodel \n'+
                              'pip3 install torch \n' +
                              'pip3 install torchvision\n'
                              'pip3 install scipy\n'+
                              'cd /home/ubuntu/openstack_torchmodel\n'+
                              'python3 run.py'
                        )
  print ('server successfully created, now wait for server booting and attach floating ip address to server')
  time.sleep(5)
  ips = connection.available_floating_ip()
  connection.add_ip_list(server,ips['floating_ip_address'])
  print ('successfully attached floating ip')
  clear_ip_cache(ips['floating_ip_address']) 
  return server,ips['floating_ip_address']

def transfer_local_file(file_name,connection,server):
#     the best way to do this is to create a container using swift, and then upload the data into the 
#     container, But the swift server is currently not installed on the cluster. I guess we can just use git then
  return 

# Because the swift is not installed on the server side, I guess just use ssh to transfer the file 
def scp(filename,ip):
  for i in range(20):
    status = os.system('scp '+filename+' ubuntu@'+ip+":/home/ubuntu/")
    time.sleep(1)
    if status ==0:
          break
    else:
          print('retry '+str(19-i)+ 'more times')
  print (status)

def main():
  filename = './identity.txt'
  conn = create_connection()
  server, ip = create_server(conn)
  print ('wait for 20 seconds for server\'s networking to initialize')
  time.sleep(20)
  scp(filename,ip)
  print ('file transferred to server')
  
main()
#conn.delete_server('testserver4')

# print(conn.available_floating_ip())
