"AN EXPERIMENTAL INVESTIGATION OF THE FLOW THROUGH AN AXIAL-FLOW PUMP"

by  Zierke, W. C., Straka, W. A., and Taylor, P. D.
    Transactions of the ASME: Journal of Fluids Engineering, Vol. 117, 
    pp. 485-490, September 1995.
 

 -- ASME JOURNAL OF FLUIDS ENGINEERING DATA BANK CONTRIBUTION HELP FILE --



1. 0  INTRODUCTION

     With a complete description of the high Reynolds number pump (HIREP)
geometry, the experimental results of Zierke, Straka, and Taylor [1993] 
providean excellent test case for comparison with numerical computations.  
This text file will briefly outline the facility, blade-row geometry, and 
locations where measurements were taken during for this HIREP experiment.  
This file will also introduce the reader to the available experimental and 
design data, as well as the format of the contributed data bank files.  All 
contributed files are in a standard ASCII form.   The reader should have 
available the detailed report by Zierke, Straka, and Taylor [1993] so that 
they may reference both the figure numbersand details such as data 
uncertainty and experimental limitations.  In order to use the provided 
experimental and design data to make valid and adequate comparisons, the 
reader must be aware of both the experimental accuracy and limitations and 
should not use the data blindly.


2.0  EXPERIMENTAL FACILITY

     This experiment was performed in the Garfield Thomas Water Tunnel at
The Applied Research Laboratory of The Pennsylvania State University.  This
tunnel has a 4-foot diameter, 14-foot long test section which supports water
velocities up to 60 ft/sec and static pressures ranging from 3 to 60 psia.  
Tunnel turbulence is controlled using a honeycomb placed within the settling 
section, 9.25 feet upstream of the nine-to-one contraction nozzle, giving a 
measured axial component of the freestream turbulence intensity level to be 
0.11% +/- 0.01% with 95% confidence.  This value was measured by 
Robbins [1978] using a conical hot-film probe and based on freestream 
velocities ranging from 20 to 60 ft/sec.  Robbins [1978] also measured a 
longitudinal integral length scale of 0.547 +/- 0.057 inches with 
95% confidence--over the same velocity range.

     HIREP consists of a 42-inch diameter pump stage driven by a 48-inch
diameter downstream turbine.  The hub has a constant diameter of 21 inches
which results in an annular area through the pump of A = 7.22 ft^2.  The pump
and turbine rotate together on a common shaft in the test section of the 
water tunnel--such that the main drive impeller of the tunnel overcomes 
the energy loses within HIREP.  The pump stage includes a row of thirteen 
inlet guide vanes, a row of seven rotating blades, and three downstream 
support struts. The turbine stage includes a row of variable pitch control 
vanes, a row of rotating blades, and a downstream cruciform support strut.

     The HIREP experiment included blade designs which are described in
detail by Zierke, Straka, and Taylor [1993]. However for both review and
emphasis, some of the important geometric characteristic should be noted 
again. To provide an experimental flow field in which one might expect better
comparisons with numerical computations, both the inlet guide vanes and rotor
blades were designed to avoid separation on the rotor blades from the 
standpoint of flow diffusion.  The rotor blades were designed for values of 
diffusion factor that increased monotonically from D = 0.12 at the tip to 
D = 0.28 at the root. Also, the moderately-loaded rotor blades were designed 
to remove the swirl at the tip and with a residual swirl which increased 
linearly towards the hub. However, in other respects--such as cavitation 
performance, head rise, flow coefficient, blade numbers, and basic 
shapes--the design was intended to represent typical axial-flow pumps.  The 
13 inlet guide vanes had a chord length of 6.90 inches and a solidity ranging 
from 1.36 at the hub to 0.68 at the tip. The 7 rotor blades had a chord 
length of 11.22 inches and a solidity of 1.19 at the hub and a chord length 
of 10.49 inches and solidity of 0.56 at the tip.  Each inlet guide vane and 
rotor blade trailing-edge geometry was essentially an asymmetric chisel, 
with the bevel of the chisel lying on the suction side of the blade and the 
point of the chisel being somewhat rounded off.  All of the inlet guide vanes 
and rotor blades included fillets where the blade sections meet the
endwalls.  The rotor blades were designed with significant negative blade
circumferential lean or skew--such that the suction surface points towards 
the hub surface.  Finally, a ring mechanism was used to allow the inlet 
guide vanes to be circumferentially indexed in 2 degree increments over 
2-1/2 blade spacings.

     At their operating design point, the rotor blades rotated clockwise
(looking upstream) at 260 rpm which yielded a rotor blade tip speed of 
U_tip = 47.6 ft/sec (using the nominal tip radius of 1.75 ft).  From 
similarity considerations, the volumetric flow coefficient was derived as

     phi = Q / (n * D_tip^3)

where Q is the volume flow rate, n is the rotor shaft speed in revolutions 
per second, D_tip is the rotor blade diameter, and V_ref is the inlet 
reference axial velocity.  This investigation was completed at the design 
value of phi = 1.36 with a dimensionless specific speed of n_tilda = 0.89.  
All data available in the data bank were obtained at this design operating 
condition unless otherwise stated in the data file header. 

 
3.0  GEOMETRY DATA

     To allow the reader to use the experimental results of Zierke, Straka, 
and Taylor [1993] as a test case for comparison with numerical computations, 
a complete description of the geometry is necessary.  The geometry of the 
HIREP facility is therefore provided to the data bank in the three following
files: 

        FILENAME   |   DESCRIPTION OF FILE
      -------------|---------------------------------------------------------                       
      HIREP-XR.IGS | Two-dimensional x-r view of the HIREP facility including
                   | the test section, nozzle, liner, hub, and blade 
                   | locations.
      -------------|---------------------------------------------------------
      HIREPIGV.IGS | IGV three-dimensional surface definition including
                   | fillets.
      -------------|---------------------------------------------------------
      HIREPROT.IGS | Rotor blade three-dimensional surface definition
                   | including fillets.
      -----------------------------------------------------------------------
                             
                                    
Each of these files are in the Initial Graphics Exchange Specification (IGES)
version 5.2 format.  They were generated using Unigraphics CAD system
version 9.1 with a UG/IGES version 10.4 translator.  


4.0  EXPERIMENTAL TECHNIQUES AND MEASUREMENT LOCATIONS

     The large-scale HIREP facility used by Zierke, Straka, and Taylor [1993]
accommodates a variety of instrumentation in both a stationary and a rotating
frame of reference.  A high-capacity, low-noise slip ring was used for many
measurements in the rotating frame and an incremental, optical shaft encoder
provided both rotor angular positions and speed.  The water temperature was
measured in the settling section with a resistance-temperature device.  For 
the entire experiment, the temperature ranged from 75 deg F to 92 deg F, 
with an average value of about 85 deg F.  This average temperature 
yields the following average properties given below:
                         
             PROPERTY            |       VALUE
        -------------------------|-----------------------------------------                     
        density, rho             |  1.932 slugs/ft^3
        -------------------------|-----------------------------------------                     
        kinematic viscosity, nu  |  0.000008755 ft^2/sec
        -------------------------|-----------------------------------------                     
        vapor pressure, p_v      | 0.5968 psia
        -------------------------|-----------------------------------------                     
                             

     As stated previously all the experimental measurements available in the
data bank were acquired at specific values of phi.  The tunnel was also
pressurized during all testing to eliminate cavitation.  To measure tunnel
conditions, we used the combination of a wall-static pressure tap and a Kiel
probe.  The static-pressure tap was located 58.6% chord axially upstream of 
the IGV leading edge in the region of the duct where both the hub and outer 
liner are cylindrical.  This tap was used to record the reference static 
pressure, p_ref. Far upstream in the settling section of the tunnel, we 
measured a reference total pressure, p_t_ref,  with a Kiel probe.  These two 
pressures along with the measured water density were used to determine a 
reference velocity.  All the data present in the available data files are at 
the design values unless otherwise noted in the file header.  The nominal 
test conditions are as shown below.  Again, it should be noted that all 
the data available in the data bank were taken at these nominal values unless 
otherwise noted in the file header. 
                         
        Property                     |    Nominal Value 
   ----------------------------------|--------------------------------------                          
    Reference Velocity, V_ref        |    35.0 ft/sec
   ----------------------------------|--------------------------------------                          
    Volumetric Flow Coefficient, phi |    1.36
   ----------------------------------|--------------------------------------                          
    Rotor Rotational Speed           |    260 rpm
   ----------------------------------|--------------------------------------                          
    Reference Static Pressure, p_ref |    43.2 psia
   ----------------------------------|--------------------------------------                          
                             
                               
4. 1 Rotor Shaft Torque and Thrust Measurements

     Force measurements are important for determining the mechanical power
required to operate the pump at various flow conditions.  In order to measure
the rotor shaft torque measurements, we mounted two 700-ohm strain gauges to
each arm of a four-active-arm Wheatstone bridge on the inside of the rotor
shaft.  The gauge measured the shear strains in the shaft wall.  Similarly, 
to measure the rotor shaft thrust, we mounted two 240-ohm strain gauges to 
each arm of a four-active-arm Wheatstone bridge which was bonded to the 
rotating hub flange.  Data were taken for a series of flow coefficients.  
The design values of the rotor torque coefficient, K_Q, and thrust 
coefficient, K_T, at phi = 1.36 are 0.301 and 0.772, respectively.  Zierke,
Straka, and Taylor [1993] give many details with regards to the calibration 
procedure and data accuracy.  However, it should be noted here that although 
the torque measurements provided very accurate data, problems in the thrust
measurement provided less than desirable results.

4.2  Blade Static-Pressure Measurements

     In order to measure the time-averaged static-pressure distributions,
pressure taps were machined into both the inlet guide vanes and rotor blades. 
One inlet guide vane included a total of 40 pressure taps on its suction 
surface only.  The adjacent vane across the same blade passage also included
40 pressure taps, however these taps were on its pressure surface.  The
pressure taps were placed on each of the blade surfaces with a pattern of
5 spanwise rows corresponding to nominal locations of 10%, 30%, 50%, 70%, 
and 90% span. The taps were individually offset slightly from these nominal 
spanwise positions.  This was done to help protect the blades from being 
locally overstressed in a single plane and to prevent any possible 
contamination of the pressure measurements from on tap due to flow 
disturbances from the upstream tap.  In a similar fashion, a single rotor
blade passage was also instrumented to include 40 pressure taps per surface. 
As explained in detail by Zierke, Straka, and Taylor [1993], the overall
uncertainty in measuring the blade static pressures ranged from 0.06 psi 
over regions of minimal streamwise pressure gradients to 0.13 psi over
regions of large pressure gradients, as might be found at 90% span on the
rotor blade suction surface.  This overall uncertainty in pressure 
measurement would result in a uncertainty in pressure coefficient on the 
order of K_p = +/- 0.009. 

4.3  Slow-Response Pressure Probes

     Various slow-response pressure techniques--including static-pressure
taps, five-hole probes, and Kiel  probes--were used to acquire both three-
component velocity data and pressure data at various locations within HIREP. 
As was mentioned previously, the tunnel and HIREP operating conditions were
obtained using a Kiel probe/static-pressure tap combination.  

     Five-hole probes were also used to measure the static and stagnation
pressures radially across the flow duct from the outside tunnel liner to 
the hub. These five-hole probes were also calibrated and capable of 
resolving the three-components of velocity.  The five-hole probes was 
used in conjunction with a strongback support which spanned the duct from 
tunnel liner to hub. This allowed the probe itself to be transversed across 
the test section with minimal bending, vibration, or fatigue effects.  
Radial surveys were completed with a five-hole probe at locations of 37.0% 
chord upstream of the IGV leading edge and a 49.7% chord downstream of the 
IGV trailing edge.  At each of these locations, measurements were made at 
approximately 50 radial positions. Zierke, Straka, and Taylor [1993] 
completed a detailed analysis of measurement uncertainties.  To summarize, 
the uncertainty in total velocity, V_tot, with 95% confidence for the 
upstream and downstream measurements was 3.1% and 7.7%, respectively.  

     While the radial five-hole probe survey provided excellent resolution in
the radial direction, they gave poor resolution in the circumferential
direction, even using the IGV ring mechanism.  In order to improve this 
circumferential resolution, a rotating rake of five-hole probes was used. 
The rake consisted of a 21-inch diameter hub that carried six five-hole 
probes, each including splitter vanes which reduced vortex shedding and 
probe vibration.   Axially, the rake measurements were acquired at the same 
location as the five-hole probe surveys, 49.7% chord downstream of the IGV 
trailing edge.  Data were obtained for 11 radial positions. The 
circumferential resolution of these measurements was 0.373 degrees.  
Considering the probe size (0.066 inches),the spatial resolution of the 
probe measurement varies from 0.344 degrees at 4.8% span to 0.185 degrees 
at 95.2% span.  In order to adequately compare to the experimental values, 
the reader is cautioned to consider the circumferential and spatial 
resolutions of the measurements and its possible averaging effects, 
especially on the wake depths.  Again Zierke, Straka, and Taylor [1993] 
completed a detailed uncertainty analysis for the rake five-hole probe 
measurements.  To summarize, the following uncertainties with 95% confidence 
were calculated as follows: V_z = +/- 0.9 ft/sec; 
V_r, V_theta = +/- 0.3 ft/sec; p, p_t = +/- 0.06 psi.

     Finally, time-averaged total-pressure measurements were acquired 32.2%
rotor tip chord axially downstream of the rotor tip trailing edge using radial
Kiel probe surveys.  As with the radial surveys using five-holes probes, the 
Kiel probes moved within a strongback support.  Data were acquired at
approximately 40 radial locations across the duct. Data were obtained for two
circumferential positions of the inlet guide vanes and, during these
measurement, a static-pressure was taken on the outer liner wall at the same
32.2% rotor tip chord location.  The uncertainty of this pressure measurement
was C_p= +/- 0.005 with 95% confidence.

4.4  Fast-Response Pressure Probe

     A fast-response total-pressure probe was designed and used to measure
the wakes and vortices associated with the rotor blades.   Zierke, Straka, 
and Taylor [1993] give further details of the probe design, calibration, and 
operation procedures.  It should be pointed out that this probe was capable of 
continuously measuring data with a response time fast enough to resolve the 
total pressures for instantaneous angular positions of the rotor--positions 
measured with the optical encoder.   However, as with most piezoelectric 
transducers of the type used, the probe was unable to measure the mean value 
of total pressure.  It did however provide excellent measurements of the 
total-pressure variation about the mean.   As with the slow-response Kiel 
probe, the fast-response total-pressure probe moved within a strongback 
support. Measurements were taken at the same axial plane as the slow-response 
Kiel probe--32.2% chord downstream of the rotor tip trailing edge 
for approximately 40 radial locations spanning the duct.     

4.5  Laser Doppler Velocimeter

     A laser Doppler velocimeter (LDV) was used to acquire nonintrusive
measurements of the flow field upstream and downstream of the rotor blade
row.  A two-component LDV system was employed allowing measurements of
both the tunnel axial, V_z, and tangential, V_theta, velocity.  All 
measurements were made on the test section centerline through a specially 
design optical window with the IGV index set at a constant position to 
eliminate the presence of IGV wakes in the measurement volume.  The specially
designed window had an inner contour which matched that of the liner 
(i.e., radius of 21 inches) and an outer radius of 63.54 inches.  The two 
radii of curvature were required to produce a near-zero magnification 
cylindrical lens for the laser optical system. In other words, the window 
allowed both the beams for each of the two components to cross at the same 
location.  In actuality, measurements of the component cross-over locations
showed that the tangential component measurement volume tended to be slightly
closer to the outer liner.  

     The data were collected using a field point measurement method.  In 
this procedure, the measurement volume remains stationary and each LDV
measurement sample is tagged with the angular position of the rotor--via 
the optical shaft encoder and a rotating machinery resolver.  Typically, for 
every position of the LDV measurement volume, a total of 100,000 samples 
were obtained, of which approximately 50% were axial velocity samples and 
the other 50% tangential.  Radial surveys were completed at locations of 
28.6% rotor tip chord upstream of the rotor tip leading edge as well as 
at 4.8%, 21.4%, and 32.2% rotor tip chord downstream of the rotor tip 
trailing edge. Also, data were obtained at approximately 76.2% span with 
an axial survey upstream of the rotor leading edge and downstream of the 
rotor trailing edge.

     The files available in the data bank include both the circumferentially-
averaged and "binned" (i.e., encoded into discrete storage windows of rotor
position) velocity data.  The reader is cautioned when using either type of
data for comparisons to numerical result.  It should remembered that all 
the LDV data was obtained at a single index position of the inlet guide 
vanes which was chosen to eliminate the presence of IGV wakes in the 
measurement volume.  As a result, the averaged data will not account for 
the effects of variations across the IGV passages and may tend to give 
slightly higher values than an actual circumferential average.  Similarly, 
the circumferentially binned data do not contain the effects of IGV wakes, 
although they are tagged with a circumferential position or bin.  The reader 
must remember that this bin position is in reality just a tag of the relative
position of the rotor and does not indicate the location relative to any 
stationary structure, including the inlet guide vanes.  Finally, when using 
the circumferentially binned data, the reader must also remember that each 
bin has some finite dimension (either 0.5 or 1.0 degrees) and, as with the
five-hole probe rake data, possible spatial averaging does occur, especially
in the wake depths.  Note that the size of the LDV measurement volume is 
smaller than the bin size.

     The data included in the data bank include the location of the
measurement, the mean velocity, and many other statistical values such as
deterministic unsteadiness.  For an explanation of this statistical analysis, 
the reader is again asked to refer to the descriptions by Zierke, Straka, 
and Taylor [1993].  That report also contains some information on both the 
precision and bias errors of the LDV measurements.  The precision errors 
with 95% confidence in the mean velocity measurements range from 0.1% to 
2.4%, with the larger errors corresponding to regions with high turbulence 
levels. Similarly, the precision errors in the turbulence intensities range 
from 6.0% to 8.5%.


5.0 CONTRIBUTED DATA FILES

     The majority of the experimental and design data presented by Zierke,
Straka, and Taylor [1993] has been made available as a test case for 
comparison with numerical computations or for further analysis.  To use this 
data, the reader should have available the detailed report by Zierke, 
Straka, and Taylor [1993] or the summary article by Zierke, Straka, and 
Taylor [1995] so that they may reference the figure numbers, analysis, and
details such as data uncertainty and experimental limitations.  In order to
use the provided experimental and design data to make valid and accurate 
comparisons, the reader must be aware of both the experimental accuracy 
and limitations and should not use the data blindly.

5.1  Data File Format

     Each of the files contained in the data bank, with the exception of the
geometry files previously mentioned, have been generated on a MS-DOS
compatible PC in ASCII format.  Each file contains a header with the
format/section information as shown below:  

                     DATA FILE HEADER LAYOUT


     TITLE:
     AUTHORS:
     JOURNAL PAPER FIGURE(S):
     PSU/ARL TR 93-12 FIGURE(S):
     *******************************************************************
     DATA TITLE:
     MEASUREMENT TYPE:
     MEASUREMENT LOCATION:

     NOMINAL CONDITIONS:
          .
          . 
          .
     DATA COLUMNS:
       1 -- 
       2 --
          .
          .
          .
     *******************************************************************  






     The first two entries, TITLE: and AUTHORS:, in the file header are
self explanatory and are just the title of the investigation and the authors.
These are the same in each of the data files.  The next entry, JOURNAL PAPER
FIGURE(S): shows the figures number(s) from Zierke, Straka, and Taylor
[1995] which contain the data given in the file.  Similarly, the fourth 
entry, PSU/ARL TR 93-12 FIGURE(S):, contains the figure number(s) from 
Zierke, Straka, and Taylor [1993] in which the data were used.

     The next section of the header is separated by a line of asterisks.  
The fifth entry, DATA TITLE, gives the title of the data file and
MEASUREMENT TYPE briefly explains the method by which the data was
obtained or calculated.  In MEASUREMENT LOCATION the location or
position where the data acquired at is given.  This is typically given 
in a percent chord from some known geometric feature such at a blade leading
edge.  The next entry, NOMINAL CONDITION, contain multiple lines of test 
conditions used when obtaining the data or parameters necessary for 
understanding the data such as geometric dimensions. Information such as 
the reference velocity, water density, rotor diameter, and annulus flow 
area are given.  The final entry in the file header, DATA COLUMNS, describes 
the data for each of the columns in the data files.  This section also 
explains the definition of any property given in one of the data columns.  
Finally, the header is ended with a line of asterisks and is followed by 
columns of numerical data which match the number columns which appears 
in the DATA COLUMNS header entry. 

     
5.2  Data Files

     The experimental and design data contained in this data bank is 
supplied in a total of 157 files totaling over 4.8 megabytes.  The geometry 
files mentioned earlier total over 2.1 megabytes and are given in 3 IGES 
files.  To allow easier access to specific parts of the database, a data 
file/figure cross-reference is supplied in the file FILELIST.TXT.  The file 
contains the filenames for all the files that are included in the HIREP 
data bank.  These files are referenced with the figure numbers from both 
references (Zierke, Straka, and Taylor [1993,1995]).  


6.0  CONCLUSIONS

The HIREP experimental measurements provide an excellent database for
comparisons with three-dimensional, turbulent, incompressible flow
computations as long as the experimental limitation are taken into account.
The database will also provide excellent empirical data for both engineering 
model formulation and validation.  Most of the experimental results and 
design calculations from the axial-flow pump experiment conducted and 
analyzed by Zierke, Straka, and Taylor [1993] have been added to the ASME 
Journal of Fluids Engineering data bank.  The data bank contributions also 
include the geometry definition contained in three IGES standard files.  
The contributed data include various measurements acquired upstream, between, 
and downstream of a two blade row axial-flow pump stage.  These data were
obtained using force cells, blade static-pressure taps, slow-response 
pressure probes, a fast-response pressure probe, and laser Doppler 
velocimetry.          

7.0 REFERENCES

Robbins, B. E., "Water Tunnel Turbulence Measurements behind a
Honeycomb," Journal of Hydronautics, Vol. 12, pp. 112-128, 1978.

Zierke, W. C., Farrell, K. J., and Straka, W. A., "Measurements of the Tip
Clearance Flow for a High Reynolds Number Axial-Flow Rotor," Transactions 
of the ASME: Journal of Turbomachinery, Vol. 117, October 1995.

Zierke, W. C., and Straka, W. A., "Surface Flow Visualization and the 
Three-Dimensional Flow in an Axial-Flow Pump," AIAA Journal of Propulsion 
and Power.

Zierke, W. C., Straka, W. A., and Taylor, P. D., "The High Reynolds Number
Flow Through an Axial-Flow Pump," Penn State/Applied Research Laboratory 
Technical Report, TR 93-12, November 1993.

Zierke, W. C., Straka, W. A., and Taylor, P. D., "An Experimental 
Investigation of the Flow through an Axial-Flow Pump," Transactions of the
ASME: Journal of Fluids Engineering, Vol. 117, pp. 485-490, September 1995.





