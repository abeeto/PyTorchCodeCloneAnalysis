
def addtoinventory (inventory,addeditems):
    for i in addeditems:
        if i not in inventory.keys():
            inventory[i]= 1
        else:
            inventory[i]+=1
    
def displayInventory(inventory):

    print("Inventory. ")
    item_total = 0

    for k,v in inventory.items():
        print(str(v) + '  ' + k)
        item_total += v
    print("Total number of items: " + str(item_total))
    print("\n")

inv = {'gold coin': 42, 'rope': 1}
dragonLoot = ['gold coin','dagger','gold coin','gold coin','ruby']

addtoinventory (inv,dragonLoot)
displayInventory(inv)
