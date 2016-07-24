
src = '/nfs/hn38/users/xiaolonw/COCO/coco-master/names_gen.txt';
src2 = '/nfs/hn38/users/xiaolonw/COCO/coco-master/names_str.txt';

fid = fopen(src, 'r');
fid2 = fopen(src2, 'w');

longs = [];

for i = 1 : 80
	num = fscanf(fid, '%d', 1);
	s = fscanf(fid, '%s', 1);
	s = ['"' s '", '];
	longs = [longs s]; 
	fprintf('%s', longs); 

end


fclose(fid); 
fclose(fid2); 

