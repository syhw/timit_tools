import pylab as pl
import numpy as np

"""
	Electropalatograph Frames *.epg, 
		62bit padded to 64bit, Raw binary. 
		Frame rate of 200 frames per second. Each
		frame consists of 8x8bit words. Each bit of each word
		represents the on/off status of each contact in the
		palatogram. The first word represents, left to right, the
		front row of the palatogram (bits 0 and 7 are unused), 
		the last word represents the back row.
"""

#fname = sys.argv[1]
fname = '/Users/gabrielsynnaeve/postdoc/datasets/MOCHA_TIMIT/msak0/msak0_001.epg'
with open(fname, 'rb') as f:



# TODO
# here is some matlab code to help
#fid = fopen(filename,'r'); 
#dummy = fread(fid,'short'); # CARE, file.read specify the total size, not the chunk size
#nbits = 16 * length(dummy);
#fprintf(1,'nb. of bits: %d\n',nbits);
#frewind(fid);
#[epg,count] = fread(fid,nbits,'ubit1');
#if count ~= nbits
#fprintf(1,'Warning: %d bits read for %s.epg.raw\n',count,filename);
#end
#nbits = count;
#nframes = nbits / 64;
#epg = reshape(epg,64,nframes);
#epg = fliplr(epg');
#epg(:,[1 8]) = [];
