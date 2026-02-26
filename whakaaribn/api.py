from datetime import timedelta, datetime
import io
import json
from typing import Optional

from tonik.api import TonikAPI
import numpy as np
import pandas as pd
from starlette.responses import StreamingResponse
from fastapi.responses import StreamingResponse
import uvicorn

from whakaaribn import convert_probability, get_data


# In an async context
async def get_df_from_streaming_response(response: StreamingResponse) -> pd.DataFrame:
    # Read all content
    body = ""
    async for chunk in response.body_iterator:
        body += chunk

    # Convert to DataFrame using StringIO
    df = pd.read_csv(io.StringIO(body))
    return df
 
class Forecast(TonikAPI):
    def __init__(self, rootdir):
        super().__init__(rootdir) # Give a name to the model
        self.app.get("/forecast")(self.forecast)
        self.app.get("/labels")(self.labels)
    
    async def forecast(self, name: str, starttime: str, endtime: str, horizon: int = 91):
        probs = await self.feature(group='whakaari_forecasts',
                                   name=name,
                                   starttime=starttime,
                                   endtime=endtime)
        probs = await get_df_from_streaming_response(probs)                   
        probs['feature'] = convert_probability(probs['feature'].interpolate().values,
                                               40, horizon)
        output = probs.to_csv(index=False, columns=['dates', 'feature'])
        return StreamingResponse(iter([output]), 
                                 media_type='text/csv',
                                 headers={'Content-Disposition': 'attachment; filename="feature.csv"',
                                 'Content-Length': str(len(output))})
    async def labels(self, starttime: Optional[str] = None, endtime: Optional[str] = None):
        if starttime is not None:
            _st = self.preprocess_datetime(starttime)
        if endtime is not None:
            _et = self.preprocess_datetime(endtime)
        labelfile = get_data('data/whakaari_labels.json')
        with open(labelfile, 'r') as f:
            labels = json.load(f)
        new_labels = []
        for label in labels:
            ntime = np.datetime64(label['time']).astype('datetime64[ms]') 
            if starttime is not None and endtime is not None:
                if ntime < _st or ntime > _et:
                    continue
            label['time'] = int(ntime.astype(int))
            try:
                ntimeEnd = np.datetime64(label['timeEnd']).astype('datetime64[ms]').astype(int) 
                label['timeEnd'] = int(ntimeEnd)
            except KeyError:
                pass
            new_labels.append(label)
        return new_labels


def main(argv=None):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--rootdir", default="/tmp")
    args = parser.parse_args(argv)
    fc = Forecast(args.rootdir)
    uvicorn.run(fc.app, host="0.0.0.0", port=8003)

if __name__ == "__main__":
    main()