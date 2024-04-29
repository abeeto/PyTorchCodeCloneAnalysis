import faust
import logging
import random
log = logging.getLogger(__name__)

class Image_only(faust.Record):
    image: str
    
app = faust.App('schp_app', broker='kafka')

schp_in = app.topic('schp_in', value_type=Image_only)
schp_out = app.topic('schp_out', value_type=Image_only)

@app.agent(schp_in, sink=[schp_out])
async def solve(data):
    async for d in data:
        if d is not None:
            print(f'Received {d}')
            yield Image_only(f"You sent me this? {d}")
        else:
            log.info("No data received")

app.main()