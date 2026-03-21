;+
; NAME:
;     mc_gethdrinfo
;
; PURPOSE:
;     Extracts requested hdr info from a FITS header.
;
; CATEGORY:
;     File I/O
;
; CALLING SEQUENCE:
;     result = mc_gethdrinfo(hdr,[keywords],CANCEL=cancel)
;
; INPUTS:
;     hdr - A FITS header string array.
;
; OPTIONAL INPUTS:
;     keywords - A string array of keywords to extract.
;
; KEYWORD PARAMETERS:
;     STRING        - If a variable is given, gethdrinfo will return a string
;                     array of the values of the keywords.
;     IGNOREMISSING - If set, then a requested keyword not in the FITS
;                     header will be skipped.  The default is the
;                     store the value as "Doesn't exist"
;     CANCEL        - Set on return if there is a problem
;    
; OUTPUTS:
;     Returns a structure with two fields, vals and coms.  The vals
;     field is itself a structure where each field is the name of the
;     keyword and the value is the value of the keyword.  The coms
;     field is itself a structure where each field is the name of the
;     keyword and the value is the comment for the keyword.
;
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;     None
;
; SIDE EFFECTS:
;     None
;
; RESTRICTIONS:
;     None
;
; PROCEDURE:
;     None
;
; EXAMPLE:
;
; MODIFICATION HISTORY:
;     2001-08-03 - Written by M. Cushing, Institute for Astronomy, UH
;     2005       - Modified to include FITS comments
;     2005-11-08 - Fixed bug where the HISTORY comments were being
;                  left as an array instead of a scalar string.
;     2017-09-18 - Major upgrade to deal with wildcard reqeusts
;-
function mc_gethdrinfo,hdr,keywords,IGNOREMISSING=ignoremissing, $
                       STRING=string,CANCEL=cancel
  cancel = 0

;  Check parameters

  if n_params() lt 1 then begin
    
     cancel = 1
     print, 'Syntax - result = mc_gethdrinfo(hdr,[keywords],STRING=string,$'
     print, '                                CANCEL=cancel)'
     return,1
     
  endif
  cancel = mc_cpar('mc_gethdrinfo',hdr,1,'Hdr',7,1)
  if cancel then return,-1

  if n_params() eq 1 then begin

;  Get keywords to extract
       
     for i = 0, n_elements(hdr)-1 do begin

        if strmid(hdr[i],8,1) eq '=' then begin
           
           key = strtrim( strmid(hdr[i],0,8),2)
           if i eq 0 then keywords = key else begin
              
              z = where(keywords eq key,count)
              if count eq 0 then keywords = (i eq 0) ? key:[keywords,key]
              
           endelse
           
        endif
        
     endfor
     
     keywords = [keywords,'HISTORY']
     
  endif else begin
     
     cancel = mc_cpar('gethdrinfo',keywords,2,'Keywords',7,[0,1])
     if cancel then return,-1

  endelse
  
  nkeys  = n_elements(keywords)

  l = 0
  for i = 0, nkeys-1 do begin

;  Search for wild card
     
     wildcard = strmatch(keywords[i],'*\*')

     if wildcard then begin

        for j = 0, n_elements(hdr)-1 do begin
           
           if strmid(hdr[j],8,1) eq '=' then begin
              
              key = strtrim(strmid(hdr[j],0,8),2)
              if strmatch(key,keywords[i]) then begin

                 val = fxpar(hdr,key,COMMENT=com)

                 vals = (l eq 0) ? create_struct(key,val): $
                        create_struct(vals,key,val)
                 coms = (l eq 0) ? create_struct(key,com): $
                        create_struct(coms,key,com)
                 string = (l eq 0) ? strtrim(val,2):[string,strtrim(val,2)]
                 l = l + 1
                 
              endif
              
           endif
           
        endfor
     
     endif else begin

        key = keywords[i]
        val = fxpar(hdr,keywords[i],COMMENT=com)
        if !err lt 0 then begin

           if keyword_set(IGNOREMISSING) then continue
           print, 'Keyword '+keywords[i]+' does not exist.'
           val = "Doesn't exist"
           com = ' ' 
                     
        endif

        vals = (l eq 0) ? create_struct(key,val):create_struct(vals,key,val)
        coms = (l eq 0) ? create_struct(key,com):create_struct(coms,key,com)
        string = (l eq 0) ? strtrim(val,2):[string,strtrim(val,2)]
        l = l + 1
                                      
     endelse
         
  endfor

  return, {vals:vals,coms:coms}     
  
end






