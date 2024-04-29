!pip install --upgrade pip
!pip install mcrcon

server_address = "your-container's FQDN is here.japaneast.azurecontainer.io"
server_passwd = "your password for Rcon"

from mcrcon import MCRcon

with MCRcon(server_address, server_passwd, 25575) as mcr:
    print(mcr.command("/time set night"))
    start_x, start_y, start_z = 6, 62, 108
    end_x, end_y, end_z  = 60, 70, 144

    for x in filter(lambda x : x % 6 == 0, range(start_x, end_x)):
        for z in filter(lambda z : z % 6 == 0, range(start_z, end_z)):
            for y in range(start_y, end_y):
                res = mcr.command("execute if block %s %s %s minecraft:air run fill %s %s %s %s %s %s minecraft:torch" % (x, y, z, x, y, z, x, y, z))
                if res == "Successfully filled 1 blocks":
                    print(res)
                    break
                res = mcr.command("execute if block %s %s %s minecraft:grass run fill %s %s %s %s %s %s minecraft:torch" % (x, y, z, x, y, z, x, y, z))
                if res == "Successfully filled 1 blocks":
                    print(res)
                    break
