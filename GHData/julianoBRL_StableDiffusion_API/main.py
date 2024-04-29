from src.services.Server import server
from src.services.SystemManager import check_essentials, download_model

#Controllers
from src.controllers.UserEndpoints import *
from src.controllers.SDEndpoints import *
from src.controllers.GalleryEndpoints import *

#Models
from src.objects.UserModel import *
from src.objects.ImageModel import *
from src.objects.JobModel import *

if __name__ == '__main__':
    check_essentials()
    download_model()
    server.run()