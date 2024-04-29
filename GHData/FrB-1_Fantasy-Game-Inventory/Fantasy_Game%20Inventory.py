#Fanstay video game.
#Inventory in an item and the value is an integer.
#function thats reads from the dict_list that will take any possible inventory and its value (an integer).
#detailing how many item the player has.

#inventory.py programme.

#dict_list
stuff = {'rope': 1,'torch':6, 'gold coin':42, 'dagger': 1, 'arrow': 12}

#function displayInventory with argument called inventory to read from dict_list.
def displayInventory (inventory):
    print("Inventory.")
    items_total = 0                     #items_total to add values of key values.
    for k,v in inventory.items():       #k to read the keys and v to read values.
        print(' '+ str(v) + '  ' + k)   #print statement.
        items_total += v                #add values integers together.

    print('Total number of items: ' + str(items_total))     #display total number of inventories.
displayInventory(stuff)                                     #calling the displayinventory function.

