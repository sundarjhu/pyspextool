;+
; NAME:
;     mc_earthvelocity
;
; PURPOSE:
;     Provides velocities of the Earth towards a celesital position in various
;     reference frames
;
; CALLING SEQUENCE:
;     result = mc_earthvelocity(yr,mo,day,hr,rah,ram,ras,decd,decm,decs,equ,$
;                               CANCEL=cancel)
;
; INPUTS:
;     yr   - Integer scalar of the year (4 digit)
;     mo   - Integer scalar of the month
;     day  - Integer scalar of the day
;     rah  - Integer scalar of the hour component of Right Ascension in
;            sexigesimal notation
;     ram  - Integer scalar of the minutes component of Right Ascension in
;            sexigesimal notation
;     ras  - Float scalar of the seconds component of Right Ascension in
;            sexigesimal notation
;     decd - Integer scalar of the degrees component of Declination in
;            sexigesimal notation
;     decm - Integer scalar of the arcminutes component of Declination
;            in sexigesimal notation
;     decs - Float scalar of the arseconds component of Declination
;            in sexigesimal notation
;     equ  - Integer scale of the equinox, e.g. 2000
;
; OPTIONAL INPUTS:
;     None
;
; KEYWORD PARAMETERS:
;     SILENT - Set to surpress reporting the results on the terminal.
;     CANCEL - Set on return if there is a problem.
;
; OUTPUTS:
;     A 3-tag structure.
;          vhelio: Velocity of Earth wrt to the sun is km s-1 towards (ra,dec)
;          vsun: Velocity of solar motion wrt LSR in km s-1 toward (ra,dec)
;          vlsr: Net radial velocity of earth wrt to LSR in km s-1
;          toward (ra,dec)
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
;     Spextool (and its dependencies)
;
; PROCEDURE:
;     Based entirely on the IDL code radvel.pro provide by William Vacca.
;
; EXAMPLES:
;     result = mc_earthvelocity(2017,03,21,0,0,0,0.,0,0,0.,2000,/SILENT)
;     result.vhelio = -0.33189940 km s-1
;     result.vsun = 0.28998249 km s-1
;     result.vlsr = -0.041916911 km s-1
;
; MODIFICATION HISTORY:
;     2017-02-18 - Written by M. Cushing, University of Toledo
;-
function mc_earthvelocity,yr,mo,day,hr,rah,ram,ras,decd,decm,decs,equ, $
                          SILENT=silent,CANCEL=cancel

  cancel = 0

  ;  Check parameters

  if n_params() lt 11 then begin
     
     print, 'Syntax - result = mc_earthvelcor(yr,mo,day,hr,rah,ram,ras,$'
     print, '                                 decd,decm,decs,equ,CANCEL=cancel)'
     cancel = 1
     return,-1
     
  endif
  
  cancel = mc_cpar('mc_earthvelcor',yr,1,'Year',[2,3],0)
  if cancel then return,-1
  cancel = mc_cpar('mc_earthvelcor',mo,2,'Month',[2,3],0)
  if cancel then return,-1
  cancel = mc_cpar('mc_earthvelcor',day,3,'Day',[2,3],0)
  if cancel then return,-1
  cancel = mc_cpar('mc_earthvelcor',hr,4,'Hour',[2,3],0)
  if cancel then return,-1
  cancel = mc_cpar('mc_earthvelcor',rah,5,'Right Ascension (Hours)',[2,3],0)
  if cancel then return,-1
  cancel = mc_cpar('mc_earthvelcor',ram,6,'Right Ascension (Minutes)',[2,3],0)
  if cancel then return,-1
  cancel = mc_cpar('mc_earthvelcor',ras,7,'Right Ascension (Seconds)',[4,5],0)
  if cancel then return,-1
  cancel = mc_cpar('mc_earthvelcor',decd,8,'Declination (Degrees)',[2,3],0)
  if cancel then return,-1
  cancel = mc_cpar('mc_earthvelcor',decm,9,'Declination (Minutes)',[2,3],0)
  if cancel then return,-1
  cancel = mc_cpar('mc_earthvelcor',decs,10,'Declination (Seconds)',[4,5],0)
  if cancel then return,-1
  cancel = mc_cpar('mc_earthvelcor',equ,11,'Equinox',[2,3,4,5],0)
  if cancel then return,-1

;--- Convert input date to JD
  jdcnv, yr, mo, day, hr, jd

;--- Compute Heliocentric (vh) and Barycentric (vb) velocity vectors
  baryvel, jd, equ, vh, vb

;--- Convert RA and Dec of object to radians
  ra  = ten(rah, ram, ras)*15.0/!RADEG
  dec = ten(decd,decm,decs)/!RADEG
  
; Compute direction cosines
  cc     = cos(dec)*cos(ra)
  cs     = cos(dec)*sin(ra)
  ss     = sin(dec)
  
;--- Compute heliocentric velocity toward object
  vhelio = vb[0]*cc + vb[1]*cs + vb[2]*ss
  
;--- Compute solar velocity wrt LSR
  v0     = 20.0                 ; solar motion wrt LSR
  ra0    = 18.0                 ; apex of solar motion to RA=18h
  dec0   = 30.0                 ; apex of solar motion to DEC=+30deg
  eq0    = 1900.0
  rasun  = ten(ra0,0.0,0.0)*15.0
  decsun = ten(dec0,0.0,0.0)
  
; Precess to current equinox
  precess, rasun, decsun, eq0, equ
  
; Convert to radians
  rasun  = rasun/!RADEG
  decsun = decsun/!RADEG
  
; Compute solar velocity in cartesian coordinates
  x0     = v0*cos(rasun)*cos(decsun)
  y0     = v0*sin(rasun)*cos(decsun)
  z0     = v0*sin(decsun)
  
; Compute the projection of the sun's motion along the line of sight
  vsun   = x0*cc + y0*cs + z0*ss
  vlsr   = vsun + vhelio
  
  if ~keyword_set(SILENT) then begin
     
     print, 'Julian Date ',jd
     print, 'Velocity of Earth wrt to the sun is ',vhelio, ' km/s toward object'
     print, 'Velocity of solar motion wrt LSR is ',vsun,' km/s toward object'
     print, 'Net radial velocity of earth wrt to LSR is ', vlsr, $
            ' km/s toward object'  
     
  endif	

  return, {vhelio:vhelio,vsun:vsun,vlsr:vlsr}
  
end
