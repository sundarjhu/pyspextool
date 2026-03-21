;+
; NAME:
;     mc_avehdrs
;
; PURPOSE:
;     Combines Spextool FITS headers together
;
; CATEGORY:
;     File I/O
;
; CALLING SEQUENCE:
;     result = mc_avehdrs(hdrs,PAIR=pair,CANCEL=cancel)
;
; INPUTS:
;     hdrs - An array [nhdrs] of structures.  Each element contains a
;            2-tag structure.
;            hdrs[0].vals = a structure with the tagname=KEYWORD, and the
;            value equal to the FITS values for the 0th hdr.
;            hdrs[0].coms = a structure with the tagname=KEYWORD, and the
;            value equal to the FITS comment for the 0th hdr.
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     PAIR   - Set if hdrs were obtained from images that were
;              actually pair subtracted.  This only affects the
;              integration time keywords.
;     CANCEL - Set on return if there is a problem
;
; OUTPUTS:
;     Returns a single 2-tag structure with the following additional
;     keywords: SRT_DATE, AVE_DATE, END_DATE, SRT_TIME, AVE_TIME,
;     END_TIME, SRT_MJD, AVE_MJD, END_MJD, SRT_HA, AVE_HA, END_HA,
;     SRT_AM, AVE_AM, END_AM.  The TOTITIME is updated.  And the
;     following keywords are removed:  DATE, TIME, MJD, HA, and AM.  
;
; OPTIONAL OUTPUTS:
;     None
;
; COMMON BLOCKS:
;     None
;
; RESTRICTIONS:
;     None
;
; DEPENDENCIES:
;     Spextool package (and its dependencies)
;
; PROCEDURE:
;     Straightforward average of dates, times, hour angles, and
;     airmasses.  Summing of integration times, and removal of
;     keywords no longer appropriate.
;     
; EXAMPLE:
;
; MODIFICATION HISTORY:
;     2001-03-30 - Written by M. Cushing, Institute for Astronomy, UH
;     2005-07-04 - Modified to accept new hdr structures from
;                  gethdrinfo
;     2012-08-06 - Added fixed keywords to be consistent with Spextool
;                  code.
;     2017-10-03 - Massive rewrite that corresponds with Spextool 5.0.
;-
function mc_avehdrs,hdrs,PAIR=pair,CANCEL=cancel

  cancel = 0

;  Check parameters

  if n_params() lt 1 then begin
     
     print, 'Syntax -  result = mc_avehdrs(hdr,PAIR=pair,CANCEL=cancel)'
     cancel = 1
     return, -1
     
  endif
  cancel = mc_cpar('mc_avehdrs',hdrs,1,'Hdrs',8 ,1)
  if cancel then return, -1
  
;  Get setup
  
  nfiles = n_elements(hdrs.vals)
  names  = tag_names(hdrs[0].vals)

  ovals = hdrs[0].vals
  ocoms = hdrs[0].coms
  
;  
;=============================== TIME and DATE  ===============================
;
;  Get the array of time values first
  
  times = dblarr(3,nfiles,/NOZERO)
  for i = 0,nfiles-1 do begin

     times[*,i] = double(strsplit(hdrs[i].vals.time,':',/EXTRACT))
     
  endfor

;  Now check to see if we observed over two days

  dates = hdrs.vals.date
  test = uniq(dates)
  
  case n_elements(test) of

     1: begin ; one day

;  Convert to decimal hours and average
        
        result = tenv(reform(times[0,*]),reform(times[1,*]),reform(times[2,*]))
        ave = total(result,/DOUBLE)/nfiles

;  Convert to sexigesimal hours

        ave = sixty(ave)

;  Write the results

        avetime = string(ave[0],FORMAT='(I2.2)')+':'+$
                  string(ave[1],FORMAT='(I2.2)')+':'+$
                  string(ave[2],FORMAT='(D09.6)')
        avedate = dates[0]
        
     end

     2: begin ; two days

;  Make the assumption that the headers are passed sequentially and
;  thus are already sorted by date

        zday2 = where(dates eq dates[test[1]],COMP=zday1)

;  Now we add 24 to the second days

        times[0,zday2] = times[0,zday2]+24

;  Convert to decimal hours and average

        result = tenv(reform(times[0,*]),reform(times[1,*]),reform(times[2,*]))
        ave = total(result,/DOUBLE)/nfiles

;  Convert to sexigesimal hours

        ave = sixty(ave)

;  Write the results
        
        if ave[0] ge 24 then begin

           avetime = string(ave[0]-24,FORMAT='(I2.2)')+':'+$
                     string(ave[1],FORMAT='(I2.2)')+':'+$
                     string(ave[2],FORMAT='(D09.6)')           

           avedate = dates[zday2[0]]
           
        endif else begin

           avetime = string(ave[0],FORMAT='(I2.2)')+':'+$
                     string(ave[1],FORMAT='(I2.2)')+':'+$
                     string(ave[2],FORMAT='(D09.6)')           

           avedate = dates[zday1[0]]           

        endelse
        
     end

     else: begin ; more than two days

        print, 'Cannot combine more than two days.'
        cancel = 1
        return, -1

     end

  endcase

;  Now add the first, ave, and last date and time

  struct_add_field, ovals, 'SRT_TIME', (hdrs.vals.time)[0], $
                    BEFORE='TIME'
  struct_add_field, ocoms, 'SRT_TIME', ' Start observation time in UTC', $
                    BEFORE='TIME'

  struct_add_field, ovals, 'AVE_TIME', avetime, $
                    BEFORE='TIME'
  struct_add_field, ocoms, 'AVE_TIME', ' Average observation time in UTC', $
                    BEFORE='TIME'  

  struct_add_field, ovals, 'END_TIME', (hdrs.vals.time)[nfiles-1], $
                    BEFORE='TIME'
  struct_add_field, ocoms, 'END_TIME', ' End observation time in UTC', $
                    BEFORE='TIME'

;  Delete the TIME tag
  
  struct_delete_field, ovals,'TIME'
  struct_delete_field, ocoms,'TIME'

   
  struct_add_field, ovals, 'SRT_DATE', (hdrs.vals.date)[0], $
                    BEFORE='DATE'
  struct_add_field, ocoms, 'SRT_DATE', ' Start observation date in UTC', $
                    BEFORE='DATE'

  struct_add_field, ovals, 'AVE_DATE', avedate, $
                    BEFORE='DATE'
  struct_add_field, ocoms, 'AVE_DATE', ' Average observation date in UTC', $
                    BEFORE='DATE'  

  struct_add_field, ovals, 'END_DATE', (hdrs.vals.date)[nfiles-1], $
                    BEFORE='DATE'
  struct_add_field, ocoms, 'END_DATE', ' End observation date in UTC', $
                    BEFORE='DATE'

  struct_delete_field, ovals,'DATE'
  struct_delete_field, ocoms,'DATE'
  
;  
;================================= MJD ===================================
;  

  ave = total(double(hdrs.vals.mjd),/DOUBLE)/double(nfiles)
  
;  Now add the first and last MJD

  struct_add_field, ovals, 'SRT_MJD', (hdrs.vals.mjd)[0],BEFORE='MJD'
  struct_add_field, ocoms, 'SRT_MJD', ' Start Modified Julian date',BEFORE='MJD'

  struct_add_field, ovals, 'AVE_MJD',string(ave,FORMAT='(D16.10)'),BEFORE='MJD'
  struct_add_field, ocoms, 'AVE_MJD', ' Average Modified Julian date',$
                    BEFORE='MJD'  

  struct_add_field, ovals, 'END_MJD', (hdrs.vals.mjd)[nfiles-1],BEFORE='MJD'
  struct_add_field, ocoms, 'END_MJD', ' End Modified Julian date',BEFORE='MJD'

  struct_delete_field, ovals,'MJD'
  struct_delete_field, ocoms,'MJD'
  
;  
;================================= HOURANG ===================================
;  

  hours = dblarr(3,nfiles,/NOZERO)
  for i = 0,nfiles-1 do begin

     
     hours[*,i] = double(strsplit(hdrs[i].vals.ha,':',/EXTRACT))
     
  endfor

;  Convert to decimal hours and average
  
  result = tenv(reform(hours[0,*]),reform(hours[1,*]),reform(hours[2,*]))  
  ave = total(result,/DOUBLE)/nfiles

;  Check for positive/negative sign
  
  pm = (ave ge 0) ? '+':'-'

;  Convert to sexigesimal hours
  
  ave = sixty(ave)

;  Write the results
  
  val = pm+string(abs(ave[0]),FORMAT='(I2.2)')+':'+$
        string(abs(ave[1]),FORMAT='(I2.2)')+':'+$
        string(abs(ave[2]),FORMAT='(F05.2)')           
  com = ' Average hour angle (hours)'


;  Now add the first and last HA

  
  struct_add_field, ovals, 'SRT_HA', (hdrs.vals.ha)[0],BEFORE='HA'
  struct_add_field, ocoms, 'SRT_HA', ' Start hour angle (hours)',BEFORE='HA'

  struct_add_field, ovals, 'AVE_HA', val, BEFORE='HA'
  struct_add_field, ocoms, 'AVE_HA', com, BEFORE='HA'  

  struct_add_field, ovals, 'END_HA', (hdrs.vals.ha)[nfiles-1],BEFORE='HA'
  struct_add_field, ocoms, 'END_HA', ' End hour angle (hours)',BEFORE='HA'  

  struct_delete_field, ovals,'HA'
  struct_delete_field, ocoms,'HA'
  
;  
;================================= AIRMASS ===================================
;  

  val = total(hdrs.vals.am)/float(nfiles)
  com = ' Average airmass'

;  Now add the first and last AM

  struct_add_field, ovals, 'SRT_AM', (hdrs.vals.am)[0],BEFORE='AM'
  struct_add_field, ocoms, 'SRT_AM', ' Start airmass',BEFORE='AM'

  struct_add_field, ovals, 'AVE_AM', val, BEFORE='AM'
  struct_add_field, ocoms, 'AVE_AM', com, BEFORE='AM'  

  struct_add_field, ovals, 'END_AM', (hdrs.vals.am)[nfiles-1],BEFORE='AM'
  struct_add_field, ocoms, 'END_AM', ' End airmass',BEFORE='AM'       

  struct_delete_field, ovals,'AM'
  struct_delete_field, ocoms,'AM'
;  
;================================= EXPTOT ===================================
;

  if keyword_set(PAIR) then begin
  
     val = hdrs[0].vals.imgitime*(nfiles/2)
     com = ' Total integration time PER BEAM (sec)'  

  endif else begin

     val = total(hdrs.vals.imgitime,/DOUBLE)
     com = ' Total integration time (sec)'  

  endelse

  struct_add_field, ovals, 'TOTITIME', val, AFTER='IMGITIME'
  struct_add_field, ocoms, 'TOTITIME', com, AFTER='IMGITIME'
  
  return, {vals:ovals,coms:ocoms}  

end

