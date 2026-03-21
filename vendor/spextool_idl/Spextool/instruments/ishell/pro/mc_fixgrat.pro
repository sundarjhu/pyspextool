pro mc_fixgrat,files,newval


  files = mc_fsextract(files,/INDEX,CANCEL=cancel)
  if cancel then return
  
  fullpaths = mc_mkfullpath('',files,/INDEX,NI=5,PREFIX='*',SUFFIX='*.fits', $
                            /EXIST,CANCEL=cancel)
  if cancel then return
     
  for i = 0,n_elements(fullpaths)-1 do begin

     ext1 = readfits(fullpaths[i],EXTEN=0,hdr1,/SILENT)
     ext2 = readfits(fullpaths[i],EXTEN=1,hdr2,/SILENT)
     ext3 = readfits(fullpaths[i],EXTEN=2,hdr3,/SILENT)

     fxaddpar,hdr1,'XDTILT',newval

     writefits,fullpaths[i],ext1,hdr1
     writefits,fullpaths[i],ext2,hdr2,/APPEND
     writefits,fullpaths[i],ext3,hdr3,/APPEND
     


  endfor
     


end
