;+
; NAME:
;     mc_wavecal1d
;
; PURPOSE:
;     To compute wavelength calibration coefficients for a single spectrum.
;
; CALLING SEQUENCE:
;     result = mc_wavecal1d(xspec,yspec,lineinfo,dispdeg,thresh,RMS=rms,$
;                           OLINEINFO=olineinfo,PLOTLINEFIND=plotlinefind,$
;                           QAPLOT=qaplot,QALINEFINDPLOT=qalinefindplot,$
;                           CANCEL=cancel)
; INPUTS:
;     xspec     - The x array (pixels).
;     yspec     - The y array (line emission spectrum).
;     lineinfo  - A structure with the following tags:
;                 lineinfo.order   = order numbers (integer)
;                 lineinfo.swave   = wavelengths (string)
;                 lineinfo.id      = line IDs (string)
;                 lineinfo.wwin    = fit window (in units of swave)
;                 lineinfo.fittype = fit type (0=gaussian, 1=Lotentzian)
;                 lineinfo.nterms  = Number of terms (3=basic,
;                                    4=basic+constant,5=basic+line) 
;                 lineinfo.xguess  = x positions of lines in wspec.
;                 lineinfo.xwin    = The fit window in units of pixels.
;      dispdeg  - The polynomial degree to fit to the line positions.
;      thresh   - The threshold over which to ignore data points int
;                 the robust polynomial fit.
;
; OPTIONAL INPUTS:
;      None
;
; KEYWORD PARAMETERS:
;      RMS          - The RMS error of the fit in units of wlines.
;      OLINEINFO    - A structure with the following tags:
;                     lineinfo.order = order numbers (integer)
;                     lineinfo.swave = wavelengths (string)
;                     lineinfo.id = line IDs (string)
;                     lineinfo.wwin = fit window (in units of swave)
;                     lineinfo.fittype = fit type (0=gaussian, 1=Lotentzian)
;                     lineinfo.nterms = Number of terms (3=basic,
;                                       4=basic+constant,5=basic+line) 
;                     lineinfo.xguess = x positions of lines in wspec.
;                     lineinfo.xwin = The fit window in units of pixels.
;                     lineinfo.xpos = the fitted position
;                     lineinfo.fwhm = the fwhm of the fit
;                     lineinfo.inten = the intensity of the line
;                     lineinfo.fnd_goodbad = The goodbad array for the
;                     line search.
;                     lineinfo.fit_goodbad = The goodbad array for the
;                     1d fit.
;     QAPLOT         - If given, the residuals QA plot with be generated.
;
;                      A structure with the following tags:
;                      qaplot.mode=A string giving the mode name.
;                      qaplot.ncols=The number of columns on the
;                      array.
;                      [qaplot.fullrange]=Set to plot the full range of
;                      residuals (including bad points).  Otherwise,
;                      the yrange is set to +-5 sigma of the good
;                      points.
;     QALINEFINDPLOT - If given, the line finding QA plot with be
;                      generated.
;                      qaplot.mode=A string giving the mode name.
;                      qaplot.loglin=Set to plot the hybrid
;                      logarithm/linear plot.
;     PLOTLINEFIND   - Set to watch the line finding.
;     CANCEL         - Set on return if there is a problem.
;
; OUTPUTS:
;     result - 1D polynomial coefficients.
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
; DEPENDENCIES:
;     Spextool library (and its dependencies)
;
; PROCEDURE:
;      Later.
;
; EXAMPLE:
;      Later.
;
; MODIFICATION HISTORY:
;      2008 - Written by M. Cushing, Institute for Astronomy,
;             University of Hawaii.
;-
function mc_wavecal1d,xspec,yspec,lineinfo,dispdeg,thresh,RMS=p2wrms, $
                      OLINEINFO=olineinfo,PLOTLINEFIND=plotlinefind, $
                      QAPLOT=qaplot,QALINEFINDPLOT=qalinefindplot, $
                      CANCEL=cancel
  
  cancel = 0

  if n_params() lt 5 then begin

     print, 'Syntax - result = mc_wavecal1d(xspec,yspec,lineinfo,dispdeg,$'
     print, '                               thresh,RMS=p2wrms, $'
     print, '                               OLINEINFO=olineinfo,$'
     print, '                               PLOTLINEFIND=plotlinefind, $'
     print, '                               QAPLOT=qaplot,$'
     print, '                               QALINEFINDPLOT=qalinefindplot,$'
     print, '                               CANCEL=cancel)'
     cancel = 1
     return, -1
     
  endif

  cancel = mc_cpar('mc_wavecal1d',xspec,1,'xspec',[2,3,4,5],[1])
  if cancel then return,-1
  cancel = mc_cpar('mc_wavecal1d',yspec,2,'yspec',[2,3,4,5],[1])
  if cancel then return,-1
  cancel = mc_cpar('mc_wavecal1d',lineinfo,3,'Lineinfo',8)
  if cancel then return,-1
  cancel = mc_cpar('mc_wavecal1d',dispdeg,5,'Dispdeg',[2,3],0)
  if cancel then return,-1
  cancel = mc_cpar('mc_wavecal1d',thresh,6,'Thresh',[2,3],0)
  if cancel then return,-1  

  nlines = n_elements(lineinfo)
  
;  Find the lines
    
  lineinfo = mc_findlines1dxd(create_struct('s1',[[xspec],[yspec]]),1, $
                              lineinfo,2,PLOTFIND=plotlinefind, $
                              QAPLOT=qalinefindplot,CANCEL=cancel)
  if cancel then return,-1
  
;  Do robust fit.

  p2wcoeffs = mc_robustpoly1d(lineinfo.xpos,double(lineinfo.swave),dispdeg, $
                              thresh,0.01,/SILENT,RMS=p2wrms, $
                              IGOODBAD=lineinfo.fnd_goodbad, $
                              OGOODBAD=p2wogoodbad,CANCEL=cancel)
  if cancel then return,-1

;  Add the goodbad array into the lineinfo structure
  
  olineinfo = lineinfo[0]
  olineinfo = create_struct(olineinfo,'fit_goodbad',0)
  olineinfo = replicate(olineinfo,nlines)
  struct_assign, lineinfo, olineinfo
  olineinfo.fit_goodbad = p2wogoodbad

;  Now do QA plot if requested

  if keyword_set(QAPLOT) then begin

     mc_mkct
     zfit = poly(lineinfo.xpos,p2wcoeffs)
     resid = (double(lineinfo.swave)-zfit)*10000.

;  Find the statistics of the residuals
     
     z = where(p2wogoodbad eq 1)
     m = moment(resid[z])
     if n_tags(qaplot) eq 3 then begin

        if ~qaplot.fullrange then $
           yrange = [m[0]-5*sqrt(m[1]),m[0]+5*sqrt(m[1])]

     endif
        
;  Find the bad/good points

     zfitbad = where(olineinfo.fit_goodbad eq 0 and $
                     olineinfo.fnd_goodbad eq 1,fitbadcnt)
     zfndbad = where(olineinfo.fnd_goodbad eq 0,fndbadcnt,$
                     NCOMP=fndgoodcnt)

     mc_setpsdev,qaplot.mode+'_1DResiduals.eps',7.5,10,FONT=14
     
     pos = mc_gridplotpos([0.1,0.1,0.95,0.95],[1,2],0.03)
  
     xmin = min(lineinfo.xpos,MAX=xmax)
     ymin = min(double(lineinfo.swave),MAX=ymax)
  

     plot,lineinfo.xpos,double(lineinfo.swave),PSYM=8,/XSTY,/YSTY, $
          POSITION=pos[*,0],FONT=0,$
          XRANGE=mc_bufrange([xmin,xmax],0.05),$
          YRANGE=mc_bufrange([ymin,ymax],0.05),/NODATA,$
          YTITLE='Wavelength (!9m!Xm)',XTICKNAME=replicate(' ',10)
     
     xx = findgen(xmax-xmin+1)+xmin
     oplot,xx,poly(xx,p2wcoeffs),COLOR=200
     
     plotsym,0,1,/FILL  
     oplot,lineinfo.xpos,double(lineinfo.swave),PSYM=8  
     plotsym,0,0.7,/FILL
     oplot,lineinfo.xpos,double(lineinfo.swave),PSYM=8,COLOR=2

     if fitbadcnt ne 0 then oplot,[lineinfo[zfitbad].xpos], $
                                  [double(lineinfo[zfitbad].swave)],$
                                  PSYM=6,SYMSIZE=1.5

     if fndbadcnt ne 0 then oplot,[lineinfo[zfndbad].xpos], $
                                  [double(lineinfo[zfndbad].swave)],$
                                  PSYM=4,SYMSIZE=1.5     
     
     del = (double(lineinfo.swave)-poly(lineinfo.xpos,p2wcoeffs))*1e4  
     ymin = min(del,MAX=ymax)
     
     plot,lineinfo.xpos,del,PSYM=8,/XSTY,/YSTY,POSITION=pos[*,1],/NOERASE, $
          FONT=0,XRANGE=mc_bufrange([xmin,xmax],0.05),$
          YRANGE=mc_bufrange([ymin,ymax],0.05),/NODATA,$
          YTITLE='Data-Model (Angstroms)',$
          XTITLE='Column (pixel)'
     
     plots,!x.crange,[0,0],LINESTYLE=1
     plotsym,0,1,/FILL  
     oplot,lineinfo.xpos,del,PSYM=8  
     plotsym,0,0.7,/FILL
     oplot,lineinfo.xpos,del,PSYM=8,COLOR=2

     if fitbadcnt ne 0 then oplot,[lineinfo[zfitbad].xpos],del[zfitbad], $
                                  PSYM=6,SYMSIZE=1.5

     if fndbadcnt ne 0 then oplot,[lineinfo[zfndbad].xpos],del[zfitbad], $
                                  PSYM=4,SYMSIZE=1.5     
     
     
     offset = 0.02
     xyouts,pos[0,0]+0.025,pos[3,0]-offset,'!9l!X Deg='+ $
            strtrim(fix(dispdeg),2), $
            FONT=0,ALIGNMENT=0,/NORM,CHARSIZE=0.9
     
     xyouts,pos[0,0]+0.025,pos[3,0]-2*offset,'RMS = '+ $
            string(sqrt(m[1]),FORMAT='(f5.3)')+' '+string(197B),$
            FONT=0,ALIGNMENT=0,/NORM,CHARSIZE=0.9
     
     xyouts,pos[0,0]+0.025,pos[3,0]-3*offset,'n!Dtot!N= '+ $
            strtrim(string(fndgoodcnt,FORMAT='(I4)'),2)+', n!Dbad!N = '+ $
            strtrim(string(fitbadcnt,FORMAT='(I2)'),2),$
            FONT=0,ALIGNMENT=0,/NORM,CHARSIZE=0.9
     
     mc_setpsdev,/CLOSE,/CONVERT,/ERASE
          
  endif

  return, p2wcoeffs

end




