from routes import torch, tf
import fastapi
import uvicorn
from fastapi.middleware import cors

# Init Main Router
app = fastapi.FastAPI()
# Setup CORS
origins = ['*']
app.add_middleware(
    cors.CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
# Add sub router (pytorch)
app.include_router(torch.router, prefix='/torch', tags=['model'])
# Add sub router (tensorflow)
app.include_router(tf.router, prefix='/tf', tags=['model'])

# Root API
@app.get('/')
async def root():
    return {
        'msg': 'Hi'
    }

# RUN SERVER
if __name__ == '__main__':
    uvicorn.run('app:app', host='127.0.0.1', port=8888, reload=True)
