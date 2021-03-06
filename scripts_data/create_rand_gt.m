
src = '/scratch/xiaolonw/nyud3/benchmarkData/gt_box_cache_dir/'

startid = 10000;
endid   = 29999;

rec.objects = [];
obj.instanceId = 1;
obj.difficult = 0;
obj.truncated = 0;
obj.bbox = [];
obj.class = 'hard_background'; 
rec.objects = [rec.objects, obj]; 
rec.imgsize = [512 512];

for i = startid : endid 
	fname = [src 'img_' num2str(i) '.mat']; 
	x1 = floor(rand() * 400) + 1;
	y1 = floor(rand() * 400) + 1; 
	w  = floor(rand() * 256) + 1; 
	h  = floor(rand() * 256) + 1; 
	x2 = min([x1 + w, 512]);
	y2 = min([y1 + h, 512]);

	bbox = [x1 y1 x2 y2];  
	rec.objects.bbox = bbox;

	save(fname, 'rec'); 

end



