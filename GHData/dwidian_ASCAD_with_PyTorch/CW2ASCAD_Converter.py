import sys
import h5py
import numpy as np
import chipwhisperer as cw

def GO_Convert(CW_format_file, ASCAD_format_file):
    proj = cw.open_project(CW_format_file)
    out_file = h5py.File(ASCAD_format_file)

    # Generate dummy mask
    mask = np.zeros(16, dtype=np.uint8)

    # Create traces dataset
    out_file.create_dataset(name="traces", data=proj.waves[:], dtype=np.array(proj.waves[0]).dtype)

    # Prepare metadata dataset
    metadata_type = np.dtype([("plaintext", proj.textins[0].dtype, (len(proj.textins[0]),)),
                              ("key", proj.keys[0].dtype, (len(proj.keys[0]),)),
                              ("masks", mask.dtype, (len(mask),))])

    traces_metadata = np.array([(proj.textins[n], proj.keys[n], mask) for n in range(proj.traces.max + 1)], dtype=metadata_type)

    # Create metadata dataset
    out_file.create_dataset("metadata", data=traces_metadata, dtype=metadata_type)

    out_file.flush()
    out_file.close()

if __name__ == "__main__":
    if len(sys.argv)!= 3 :
        print("Error: Two arguments expected")
        sys.exit(-1)
    else:
        GO_Convert(sys.argv[1], sys.argv[2])

