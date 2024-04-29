#!/usr/bin/env python
# -*- encoding: utf8 -*-
# @Author: nagexiucai
# @E-mail: me@nagexiucai.com

'''
Client for torch file system.

torchfsc[ -h]
torchfsc action [namespace[ -h]]
  action: get|put|post|delete|snap
  namespace: settings|storages
torchfsc get settings               show all key-value options
torchfsc get settings key           show value of the option named key
torchfsc put settings json          initialize key-value options with those in json and these in default
torchfsc post settings json         update key-value options with those in json
torchfsc delete settings[ a,b,c]    replace all key-value options by these in default or only the given options named a or b or c
torchfsc get storages               show all uris and each one represents an entry
torchfsc get storages uri           show details of the entry named uri partly
  torchfsc get storages host:port/any/path
  type:
  createtime:
  updatetime:
  status:
  bytes:
torchfsc get storages uri -v        show details of the entry named uri entirely
torchfsc get storages uri uri       download data of the first uri to the second uri
torchfsc put storages uri           attach an entry
torchfsc put storages uri uri       upload data of the second uri to the first uri
torchfsc post storages json         update some details in json of all entries
torchfsc post storages uri json     update some details in json of the entry named uri
torchfsc delete storages            detach all enties
torchfsc delete storages uri        detach the entry named uri
torchfsc snap[ settings|storages]   take a portrait for whole or the given namespace
'''

print locals().get("__doc__")
