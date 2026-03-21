;+
; NAME:
;     mc_findlines1dxd
;
; PURPOSE:
;     To measure the positions of emission lines in an XD spectrum.
;
; CALLING SEQUENCE:
;     result = mc_findlines1dxd(spectra,orders,lineinfo,xthresh,QAPLOT=qaplot, $
;                               PLOTFIND=plotfind,CANCEL=cancel)
;
; INPUTS:
;     spectra  - A structure with norders tags.  Each tag is the
;                spectrum for a given order:
;                wave=(spectra.(i))[*,0]
;                flux=(spectra.(i))[*,1]
;     orders   - A [norders] array of order numbers
;     lineinfo - A structure with the following tags:
;                lineinfo.order   = order numbers (integer)
;                lineinfo.swave   = wavelengths (string)
;                lineinfo.id      = line IDs (string)
;                lineinfo.wwin    = fit window (in units of swave)
;                lineinfo.fittype = fit type (0=gaussian, 1=Lotentzian)
;                lineinfo.nterms  = Number of terms (3=basic,
;                                   4=basic+constant,5=basic+line) 
;                lineinfo.xguess  = x positions of lines in wspec.
;                lineinfo.xwin    = The fit window in units of pixels.
;     xthresh  - threshold over which to ignore the fit.  If the fit
;                position, fitx, is guessx-xthresh .ge. fitx
;                .le. guessx+xthresh then the fit is good. 
;
; OPTIONAL INPUTS:
;      None
;
; KEYWORD PARAMETERS:
;     QAPLOT   - If given, the line finding QA plot with be
;                generated.
;                qaplot.mode=A string giving the mode name.
;                qaplot.loglin=Set to plot the hybrid logarithm/linear plot.
;     PLOTFIND - Set to plot the line finding interactively.
;
; OUTPUTS:
;     results - A structure with the following tags:
;               lineinfo.order   = order numbers (integer)
;               lineinfo.swave   = wavelengths (string)
;               lineinfo.id      = line IDs (string)
;               lineinfo.wwin    = fit window (in units of swave)
;               lineinfo.fittype = fit type (0=gaussian, 1=Lotentzian)
;               lineinfo.nterms  = Number of terms (3=basic,
;                                  4=basic+constant,5=basic+line) 
;               lineinfo.xguess  = x positions of lines in wspec.
;               lineinfo.xwin    = The fit window in units of pixels.
;               lineinfo.xpos    = the fitted position
;               lineinfo.fwhm    = the fwhm of the fit
;               lineinfo.inten   = the intensity of the line
;               lineinfo.fnd_goodbad = the fit goodbad array.
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
;     Spextool library (and its dependencies)
;
; PROCEDURE:
;     Fits a gaussian or lorentzian to each line using the parameters
;     passed by the user.  Optionally creates QA plots or
;     interactively fits the lines.
;
; EXAMPLES:
;     NA
;
; MODIFICATION HISTORY:
;     2017-10-01:  Written by M. Cushing, Univesrity of Toeldo
;-
pro findlines_plotfit,x,y,lineinfo,zline,yfit,a,goodbad,title,FONT=font, $
                      LOGLIN=loglin,CANCEL=cancel

  pos = mc_gridplotpos([0.15,0.1,0.95,0.95],[1,2],0.05,CANCEL=cancel)
  if cancel then return
  
;  Plot the full spectrum

;  Check for very low pixels
  
  mc_moments,y,mean,var,stddev,ROBUST=3,/SILENT
  z = where(y lt mean-10*stddev,cnt)
  if cnt ne 0 then y[z] = !values.f_nan  

;  figure out the noise level in the baseline and get yrange
  
  result = mc_robustsg(findgen(n_elements(y)),y,50,3,0.1)     
  
  mc_moments,y-result[*,1],mean,var,stddev,ROBUST=4,/SILENT
  yranges = [min(result[*,1],/NAN)-3*stddev,min(result[*,1],/NAN)+5*stddev, $
             max(y,/NAN)]
  
  if keyword_set(LOGLIN) then begin

     ysize = pos[3,0]-pos[1,0]
     lowpos = [pos[0,0],pos[1,0],pos[2,0],pos[1,0]+0.3*ysize]
     
     hipos = [pos[0,0],pos[1,0]+0.3*ysize,pos[2,0],pos[3,0]]
     
     plot,x,y,XSTY=9,/YSTY,FONT=font, POSITION=lowpos,YRANGE=yranges[0:1]

     plot,x,y,XSTY=5,/YSTY,FONT=font, POSITION=hipos,$
          /NOERASE,NODATA=0,YRANGE=[yranges[1:2]],/YLOG,$
          YTITLE='Relative Intensity',TITLE=title

     axis,XAXIS=1,/XSTY,XTICKNAME=replicate(' ',10),FONT=0


     xy = convert_coord(lineinfo.xguess,1,/DATA,/TO_NORM)

     plots,[xy[0],xy[0]],[pos[1,0],pos[3,0]],COLOR=6,/NORM
     
  endif else begin
     
     plot,x,y,/XSTY,/YSTY,FONT=font,POSITION=pos[*,0],TITLE=title, $
          YRANGE=[yranges[0],yranges[2]],YTITLE='Relative Intensity'
          
     endelse
  
  plots,[lineinfo.xguess,lineinfo.xguess],!y.crange,COLOR=6
   
;  Plot the fit
  
  ymin = min(y[zline],MAX=ymax)
  title = (goodbad eq 1) ? 'Good':'Bad'

  plot,x[zline],y[zline],/XSTY,PSYM=10,TITLE=title,XTITLE='Column (pixels)', $
       YTITLE='Arbitrary Flux',/YSTY,YRANGE=mc_bufrange([ymin,ymax],0.1), $
       POSITION=pos[*,1],FONT=font,/NOERASE
  
  oplot,x[zline],yfit,COLOR=3,PSYM=10
  plots,[lineinfo.xguess,lineinfo.xguess],!y.crange,COLOR=6
  plots,[a[1],a[1]],!y.crange,COLOR=3
  plots,[a[2]-2,a[2]-2],!y.crange,COLOR=2,LINESTYLE=1
  plots,[a[2]+2,a[2]+2],!y.crange,COLOR=2,LINESTYLE=1

end
;
;==============================================================================
;
function mc_findlines1dxd,spectra,orders,lineinfo,xthresh,QAPLOT=qaplot, $
                          PLOTFIND=plotfind,CANCEL=cancel

  cancel = 0

;  Check parameters

  if n_params() lt 4 then begin

     print, 'Syntax - mc_findlines1dxd(spectra,orders,lineinfo,xthresh,$'
     print, '                          QAPLOT=qaplot,PLOTFIND=plotfind,$'
     print, '                          CANCEL=cancel)'
     cancel = 1
     return,-1
     
  endif
  cancel = mc_cpar('mc_fitlines1dxd',spectra,1,'Spectra',8)
  if cancel then return,-1
  cancel = mc_cpar('mc_fitlines1dxd',orders,2,'Orders',[2,3],[0,1])
  if cancel then return,-1
  cancel = mc_cpar('mc_fitlines1dxd',lineinfo,3,'Lineinfo',8)
  if cancel then return,-1
  cancel = mc_cpar('mc_fitlines1dxd',xthresh,4,'Xthresh',[2,3,4,5],0)
  if cancel then return,-1      
  
;  Deal with errors (there can be underflows in mpfitpeak)

  currentExcept = !Except
  !Except = 0
  void = check_math()
  
  norders = n_elements(orders)
  nlines = n_elements(lineinfo[*].swave)  

;  Deal with plotting, real time takes precedence

  if keyword_set(QAPLOT) ne 0 then begin

     qafile = qaplot.mode+'_1DLineFind.ps'
     loglin = qaplot.loglin
     font=0
     plotsum = 1
     
  endif else plotsum = 0

  if keyword_set(PLOTFIND) ne 0 then plotsum = plotsum+2
  
;  Now set up the plotting
  
  case plotsum of

     0:

     1:  begin

        xthick=!x.thick
        ythick=!y.thick
        pthick=!p.thick
        
        !x.thick=3
        !y.thick=3
        !p.thick=3
        
        mc_mkct,0,BOT=cmapbot
        mc_setpsdev,qafile,8.5,11,FONT=13
                
     end

     else: begin ;  (2 or 3)

        mc_mkct
        window,/FREE,YSIZE=((get_screen_size())[1]-30) < 1000,XSIZE=600
        win_idx = !d.window
               
     end

  endcase
  
;  Start the "store" arrays

  lxpos = make_array(nlines,/FLOAT,VALUE=!values.f_nan)
  lfwhm = make_array(nlines,/FLOAT,VALUE=!values.f_nan)
  linten = make_array(nlines,/FLOAT,VALUE=!values.f_nan)
  lgoodbad = make_array(nlines,/BYTE,VALUE=1)
  
;  Loop over each order with lines to fit

  for i = 0,nlines-1 do begin

     zordr = where(lineinfo[i].order eq orders)

;  Get spectrum for this order and clip NaNs
     
     x = (spectra.(zordr))[*,0]
     y = (spectra.(zordr))[*,1]
     
     z = where(finite(x) eq 1)
     x = x[z]
     y = y[z]  

;  Cut the line out

     zline = where(x ge lineinfo[i].xguess-lineinfo[i].xwin/2 and $
                   x le lineinfo[i].xguess+lineinfo[i].xwin/2,linecnt)
     
     if linecnt lt lineinfo[i].nterms then begin
        
        lgoodbad[i] = 0
        lxpos[i] = !values.f_nan
        lfwhm[i] = !values.f_nan
        linten[i] = !values.f_nan
        continue
           
     endif
     
;  Determine fit type
        
     gaussian = 0
     lorentzian = 0
     case strtrim(lineinfo[i].fittype,2) of
        
        'L': lorentzian = 1
        
        'G': gaussian = 1
        
     endcase
     
;  For the case of 3 parameters, subtract off the minimum value in the
;  line region.
        
     offset = (lineinfo[i].nterms eq 3) ? min(y[zline]):0D
     yfit = mpfitpeak(x[zline],y[zline]-offset,a,NTERMS=lineinfo[i].nterms,$
                      GAUSSIAN=gaussian,LORENTZIAN=lorentzian)
     
;  Store the result

     linten[i] = a[0]
     lxpos[i] = a[1]
     lfwhm[i] = a[2]
     
;  Check to see whether it is a good fit
     
     if a[1] le lineinfo[i].xguess+xthresh and $
        a[1] ge lineinfo[i].xguess-xthresh and $
        a[2] gt 0.0 and a[0] gt 0 then lgoodbad[i]  = 1     

     if plotsum ge 1 then begin

        findlines_plotfit,x,y,lineinfo[i],zline,yfit+offset,a,lgoodbad[i], $
                          'Order '+strtrim(lineinfo[i].order,2)+ $
                          ', Wavelength='+strtrim(lineinfo[i].swave,2), $
                          FONT=font,LOGLIN=1,CANCEL=cancel
        if cancel then return, -1

        if plotsum gt 1 then begin

           cancel = mc_pause()
           if cancel then return,-1
                              
        endif
        
     endif        

  endfor
  
;  Add these arrays onto the lineinfo structure
  
  olineinfo = lineinfo[0]
  olineinfo = create_struct(olineinfo,'xpos',0.0,'fwhm',0.0,'inten',0.0, $
                            'fnd_goodbad',0)
  olineinfo = replicate(olineinfo,nlines)
  struct_assign, lineinfo, olineinfo
  
  olineinfo.xpos = lxpos
  olineinfo.fwhm = lfwhm
  olineinfo.inten = linten
  olineinfo.fnd_goodbad = lgoodbad

;  Deal with plotting stuff
  
  if plotsum gt 1 then begin
     
     wdelete, win_idx
     !p.multi = 0

  endif

  if plotsum eq 1 then begin

     mc_setpsdev,/CLOSE,/CONVERT,/ERASE
     
     !x.thick=xthick
     !y.thick=ythick
     !p.thick=pthick
          
  endif
     
;  Deal with errors
  
  status = Check_Math() ; Get status and reset accumulated math error register.
  IF(status AND NOT 32 ) NE 0 THEN $
     Message, 'IDL Check_Math() error: ' + StrTrim(status, 2)
  !Except = currentExcept
  
  return, olineinfo
  
end

