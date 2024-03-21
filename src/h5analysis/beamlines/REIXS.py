from h5analysis.LoadData import *
from h5analysis.MathData import *
from h5analysis.config import h5Config

class PlotLoader(Load1d):
    def compare(self,plot_object):
        for i in range(len(plot_object.data)):
            PlotLoader.loadObj(self, plot_object, i)
        return PlotLoader
    def plot(self, **kwargs):
        total_xlabel = ''
        for i in range(len(self.data)):
            for j in self.data[i].keys():
                if self.data[i][j].xlabel not in total_xlabel:
                    if len(total_xlabel) > 0:
                         total_xlabel += '/ '
                    total_xlabel += self.data[i][j].xlabel + ' '
        return Load1d.plot(self, xlabel = total_xlabel, ylabel = 'Value', plot_width = 1200, plot_height = 800,**kwargs)
        
class BeamlineLoader(LoadBeamline):
    def plot(self, **kwargs):
        return LoadBeamline.plot(self, xlabel ='Scan #', ylabel='Value', plot_width = 1200, plot_height = 800,**kwargs)
    def compare(self,plot_object):
        for i in range(len(plot_object.data)):
            BeamlineLoader.loadObj(self, plot_object, i)
        return BeamlineLoader
        
class SimpleLoader(Load1d):
    def load(self,config,file,y_stream,*args,**kwargs):
        return Load1d.load(self,config,file,'',y_stream,*args,**kwargs)
    def plot(self, **kwargs):
        return Load1d.plot(self, xlabel ='Point #', ylabel='Value', plot_width = 1200, plot_height = 800,**kwargs)
    def compare(self,plot_object):
        for i in range(len(plot_object.data)):
            SimpleLoader.loadObj(self, plot_object, i)
        return SimpleLoader

class XESLoader(Load1d):
    def load(self,config,file,y_stream,*args,**kwargs):
        data_args = y_stream.split('[')
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            x_stream = 'Energy[' + data_args2[0] + ']'
            y_stream = data_args[0]+data_args2[1]
        else:
            x_stream = '[None]'
        return Load1d.load(self,config,file,x_stream,y_stream,*args,**kwargs)
    def add(self,config,file,y_stream,*args,**kwargs):
        data_args = y_stream.split('[')
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            x_stream = 'Energy[' + data_args2[0] + ']'
            y_stream = data_args[0]+data_args2[1]
            
        else:
            x_stream = '[None]'
        return Load1d.add(self,config,file,x_stream,y_stream,*args,**kwargs)
    def subtract(self,config,file,y_stream,*args,**kwargs):
        data_args = y_stream.split('[')
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            x_stream = 'Energy[' + data_args2[0] + ']'
            y_stream = data_args[0]+data_args2[1]
        else:
            x_stream = '[None]'
        return Load1d.subtract(self,config,file,x_stream,y_stream,*args,**kwargs)
    def background(self,config,file,y_stream,*args,**kwargs):
        data_args = y_stream.split('[')
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            x_stream = 'Energy[' + data_args2[0] + ']'
            y_stream = data_args[0]+data_args2[1]
        else:
            x_stream = '[None]'
        return Load1d.background(self,config,file,x_stream,y_stream,*args,**kwargs)
    def compare(self,plot_object):
        for i in range(len(plot_object.data)):
            XESLoader.loadObj(self, plot_object, i)
        return XESLoader
    def plot(self, **kwargs):
        return Load1d.plot(self, x_axis_label='Emission Energy [eV]', y_axis_label='Counts', plot_width = 1200, plot_height = 800,**kwargs)
        
class XEOLLoader(Load1d):
    def load(self,config,file,y_stream,*args,**kwargs):
        data_args = y_stream.split('[')
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            x_stream = 'Energy[' + data_args2[0] + ']'
            y_stream = data_args[0]+data_args2[1]
        else:
            x_stream = '[None]' 
        return Load1d.load(self,config,file,x_stream,y_stream,*args,**kwargs)
    def add(self,config,file,y_stream,*args,**kwargs):
        data_args = y_stream.split('[')
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            x_stream = 'Energy[' + data_args2[0] + ']'
            y_stream = data_args[0]+data_args2[1]
        else:
            x_stream = '[None]' 
        return Load1d.add(self,config,file,x_stream,y_stream,*args,**kwargs)
    def subtract(self,config,file,y_stream,*args,**kwargs):
        data_args = y_stream.split('[')
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            x_stream = 'Energy[' + data_args2[0] + ']'
            y_stream = data_args[0]+data_args2[1]
        else:
            x_stream = '[None]' 
        return Load1d.subtract(self,config,file,x_stream,y_stream,*args,**kwargs)
    def background(self,config,file,y_stream,*args,**kwargs):
        data_args = y_stream.split('[')
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            x_stream = 'Energy[' + data_args2[0] + ']'
            y_stream = data_args[0]+data_args2[1]
        else:
            x_stream = '[None]' 
        return Load1d.background(self,config,file,x_stream,y_stream,*args,**kwargs)
    def plot(self, **kwargs):
        XESLoader.plot_legend(self,'top_left')
        return Load1d.plot(self, x_axis_label='Wavelength [nm]', y_axis_label='Counts', plot_width = 1200, plot_height = 800,**kwargs)
   
    
class XASLoader(Load1d):
    def load(self,config,file,y_stream,*args,**kwargs):
        return Load1d.load(self,config,file,'Energy',y_stream,*args,**kwargs)
    def add(self,config,file,y_stream,*args,**kwargs):
        return Load1d.add(self,config,file,'Energy',y_stream,*args,**kwargs)
    def subtract(self,config,file,y_stream,*args,**kwargs):
        return Load1d.subtract(self,config,file,'Energy',y_stream,*args,**kwargs)
    def background(self,config,file,y_stream,*args,**kwargs):
        return Load1d.background(self,config,file,'Energy',y_stream,*args,**kwargs)
    def compare(self,plot_object):
        for i in range(len(plot_object.data)):
            XASLoader.loadObj(self, plot_object, i)
        return XASLoader
    def plot(self, **kwargs):
        return Load1d.plot(self, x_axis_label='Excitation Energy [eV]', y_axis_label='Intensity', plot_width = 1200, plot_height = 800,**kwargs)
    
class PFYLoader(Object2dReduce):
    def load(self,config,file, y_stream,*args, **kwargs):
        PFY_Image = Load2d()
        data_args = y_stream.split('[')
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            y_stream_mod = data_args[0] + data_args2[1]
        PFY_Image.load(config,file,'Energy',y_stream_mod,*args,**kwargs)
        if(len(data_args2) > 1):
            data_args3 = data_args2[0].split(',')
            y_range1 = data_args3[0].split(':')
            y_min1 = float(min(y_range1))
            y_max1 = float(max(y_range1))
            x_min = float(min(PFY_Image.data[0][args[0]].new_x))
            y_range2 = data_args3[1].split(':')
            y_min2 = float(min(y_range2))
            y_max2 = float(max(y_range2))
            x_max = float(max(PFY_Image.data[0][args[0]].new_x))
        Object2dReduce.load(self, PFY_Image,0,*args)
        Object2dReduce.polygon(self, 'y', [(x_min,y_min1),(x_min,y_max1),(x_max, y_max2),(x_max,y_min2)], exact = False)
        self.data[len(self.data)-1][0].legend = 'S' + str(args[0]) + str(y_stream)
        return Object2dReduce
    def add(self,config,file, y_stream,*args, **kwargs):
        PFY_Image = Load2d()
        data_args = y_stream.split('[')
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            y_stream_mod = data_args[0] + data_args2[1]
        PFY_Image.add(config,file,'Energy',y_stream_mod,*args,**kwargs)
        if(len(data_args2) > 1):
            data_args3 = data_args2[0].split(',')
            y_range1 = data_args3[0].split(':')
            y_min1 = float(min(y_range1))
            y_max1 = float(max(y_range1))
            x_min = float(min(PFY_Image.data[0][0].new_x))
            y_range2 = data_args3[1].split(':')
            y_min2 = float(min(y_range2))
            y_max2 = float(max(y_range2))
            x_max = float(max(PFY_Image.data[0][0].new_x))
        #PFY_Spectrum = Object2dReduce()
        Object2dReduce.load(self, PFY_Image,0,0)
        Object2dReduce.polygon(self, 'y', [(x_min,y_min1),(x_min,y_max1),(x_max, y_max2),(x_max,y_min2)], exact = False)
        self.data[len(self.data)-1][0].legend = 'S' + str(args[0])
        for i in range(1, len(args)):
            self.data[len(self.data)-1][0].legend += '+' + str(args[i])
        self.data[len(self.data)-1][0].legend += str(y_stream)
        return Object2dReduce
    def subtract(self,config,file, y_stream,*args, **kwargs):
        PFY_Image = Load2d()
        data_args = y_stream.split('[')
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            y_stream_mod = data_args[0] + data_args2[1]
        PFY_Image.subtract(config,file,'Energy',y_stream_mod,*args,**kwargs)
        if(len(data_args2) > 1):
            data_args3 = data_args2[0].split(',')
            y_range1 = data_args3[0].split(':')
            y_min1 = float(min(y_range1))
            y_max1 = float(max(y_range1))
            x_min = float(min(PFY_Image.data[0][0].new_x))
            y_range2 = data_args3[1].split(':')
            y_min2 = float(min(y_range2))
            y_max2 = float(max(y_range2))
            x_max = float(max(PFY_Image.data[0][0].new_x))
        #PFY_Spectrum = Object2dReduce()
        Object2dReduce.load(self, PFY_Image,0,0)
        Object2dReduce.polygon(self, 'y', [(x_min,y_min1),(x_min,y_max1),(x_max, y_max2),(x_max,y_min2)], exact = False)
        self.data[len(self.data)-1][0].legend = 'S' + str(args[0])
        for i in range(1, len(args)):
            self.data[len(self.data)-1][0].legend += '-' + str(args[i])
        self.data[len(self.data)-1][0].legend += str(y_stream)
        return Object2dReduce
    def compare(self,plot_object):
        for i in range(len(plot_object.data)):
            PFYLoader.loadObj(self, plot_object, i)
        return PFYLoader
    def plot(self, x_axis_label='Excitation Energy [eV]', y_axis_label='Intensity', plot_width = 1200, plot_height = 800, **kwargs):
        return Object2dReduce.plot(self, x_axis_label=x_axis_label, y_axis_label=y_axis_label, plot_width = plot_width, plot_height = plot_height, **kwargs)

class EEMSLoader(Load2d):
    def load(self,config,file,y_stream,*args,**kwargs):
        return Load2d.load(self,config,file,'Energy',y_stream,*args,**kwargs)
    def add(self,config,file,y_stream,*args,**kwargs):
        return Load2d.add(self,config,file,'Energy',y_stream,*args,**kwargs)
    def subtract(self,config,file,y_stream,*args,**kwargs):
        return Load2d.subtract(self,config,file,'Energy',y_stream,*args,**kwargs)
    def background(self,config,file,y_stream,*args,**kwargs):
        return Load2d.background(self,config,file,'Energy',y_stream,*args,**kwargs)
    def plot(self, **kwargs):
        return Load2d.plot(self, xlabel='Excitation Energy [eV]', ylabel='Emission Energy [eV]', plot_width = 1200, plot_height = 800,**kwargs)
        
class MESHLoader(LoadHistogram):
    def plot(self, **kwargs):
        return LoadHistogram.plot(self, plot_width = 1200, plot_height = 800,**kwargs)
 
class RIXSMapper(Object2dTransform):
    def load(self,config,file,y_stream,*args,**kwargs):
        data_args = y_stream.split('[')
        y_stream = data_args[0]
        Object2dTransform.load(self,config,file,'Energy',y_stream,*args,**kwargs)
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            data_args3 = data_args2[0].split(':')
            x_range = (float(data_args3[0]), float(data_args3[1]))
            Object2dTransform.transform(self,'x-y', xlim = x_range)
        else:
            Object2dTransform.transform(self,'x-y')
        return Object2dTransform.transpose(self)
    
class ELOSSLoader(Object2dReduce):
    def load(self,config,file, y_stream,*args,**kwargs):
        data_args = y_stream.split('[')
        y_stream = data_args[0]
        RIXS_Image = Object2dTransform()
        RIXS_Image.load(config,file,'Energy',y_stream,*args,**kwargs)
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            data_args3 = data_args2[0].split(':')
            x_range = (float(data_args3[0]), float(data_args3[1]))
            RIXS_Image.transform('x-y', xlim = x_range)
        else:
            RIXS_Image.transform('x-y')    
        RIXS_Image.transpose()
        Object2dReduce.load(self, RIXS_Image,0,*args)
        return Object2dReduce.roi(self, 'y', roi =  x_range)
    
class ETLoader(Object2dReduce):
    def load(self,config,file, y_stream,*args,**kwargs):
        data_args = y_stream.split('[')
        y_stream = data_args[0]
        RIXS_Image = Object2dTransform()
        RIXS_Image.load(config,file,'Energy',y_stream,*args,**kwargs)
        if(len(data_args) > 1):
            data_args2 = data_args[1].split(']')
            data_args3 = data_args2[0].split(':')
            y_range = (float(data_args3[0]), float(data_args3[1]))
            RIXS_Image.transform('x-y', ylim = y_range)
        else:
            RIXS_Image.transform('x-y')    
        RIXS_Image.transpose()
        Object2dReduce.load(self, RIXS_Image,0,*args)
        return Object2dReduce.roi(self, 'x', roi =  y_range)
    
    
class MCPLoader(Load2d):
    def load(self,config,file,y_stream,*args,**kwargs):
        y_stream = y_stream + '[None:None,None:None]'
        x_stream = '[None]'
        return Load2d.load(self,config,file,x_stream,y_stream,*args,**kwargs)
    def add(self,config,file,y_stream,*args,**kwargs):
        y_stream = y_stream + '[None:None,None:None]'
        x_stream = '[None]'
        return Load2d.add(self,config,file,x_stream,y_stream,*args,**kwargs)
    def subtract(self,config,file,y_stream,*args,**kwargs):
        y_stream = y_stream + '[None:None,None:None]'
        x_stream = '[None]'
        return Load2d.subtract(self,config,file,x_stream,y_stream,*args,**kwargs)
    

#RSXS ES Configuration#
RSXS = h5Config()
RSXS.sca_folder('Data')
RSXS.key("SCAN_{scan:03d}",'scan')

#Aliases
RSXS.sca('Energy','Data/beam')
RSXS.sca('TEY','Data/tey')
RSXS.sca('I0','Data/i0')
RSXS.sca('TEY_N','Data/tey', norm_by = 'Data/i0')

#MCA Detectors
RSXS.mca('SDDX','Data/sdd_xrf_mca','Data/sdd_xrf_scale',None)
RSXS.mca('SDDA','Data/sdd_arm_mca','Data/sdd_arm_scale',None)
RSXS.mca('SDDX_N','Data/sdd_xrf_mca','Data/sdd_xrf_scale','Data/i0')
RSXS.mca('SDDA_N','Data/sdd_arm_mca','Data/sdd_arm_scale','Data/i0')

#IMAGE Detectors
RSXS.stack('mcpIMG','Data/mcp_a_img','Data/mcp_tth_scale','Data/mcp_detz_scale',None)



#Log Book for RIXS ES
rsxs_log = dict()
rsxs_log['Command'] = 'command'
rsxs_log['Sample'] = 'Endstation/Sample/Name'
rsxs_log['Comments'] = ('comment_01','comment_02','comment_03','comment_04','comment_05','comment_06','comment_07','comment_08','comment_09','comment_10')
rsxs_log['X'] = ['Endstation/Motors/x', 3]
rsxs_log['Y'] = ['Endstation/Motors/y', 3]
rsxs_log['Z'] = ['Endstation/Motors/z', 3]
rsxs_log['Theta'] = ['Endstation/Motors/th', 3]
rsxs_log['2Theta'] = ['Endstation/Motors/tth', 3]
rsxs_log['Chi'] = ['Endstation/Motors/chi',3]
rsxs_log['Phi'] = ['Endstation/Motors/phi',3]
rsxs_log['Detz'] = ['Endstation/Motors/detz',3]
rsxs_log['Temperature'] = ['Endstation/Counters/t_k', 2]
rsxs_log['Energy'] = ['Beamline/Monochromator/beam',2]
rsxs_log['Exit Slit'] = ['Beamline/Apertures/Exit_Slit/vert_gap',1]
rsxs_log['Flux'] = 'Beamline/flux'
rsxs_log['Dwell'] = ['Endstation/Counters/sec', 1]
rsxs_log['Mirror/Grating'] = ('/Beamline/Monochromator/grating','/Beamline/Monochromator/mirror')
rsxs_log['Polar/Harmonic'] = ('Beamline/Source/EPU/polarization', 'Beamline/Source/EPU/harmonic')
rsxs_log['Status'] = 'status'
rsxs_log['Date'] = 'date'

#Data Structure Config
RIXS = h5Config()
RIXS.sca_folder('Data')
RIXS.key("SCAN_{scan:03d}",'scan')

#Aliases
RIXS.sca('Energy','Data/beam')
RIXS.sca('TEY','Data/tey')
RIXS.sca('I0','Data/i0')
RIXS.sca('TEY_N','Data/tey', norm_by = 'Data/i0')

#MCA Detectors
RIXS.mca('SDDA','Data/sdd_a_mca','Data/sdd_a_scale',None)
RIXS.mca('SDDA_NOSCALE','Data/sdd_a_mca',None,None)
RIXS.mca('SDDA_N','Data/sdd_a_mca','Data/sdd_a_scale',norm_by = 'Data/i0')
RIXS.mca('SDDB','Data/sdd_b_mca','Data/sdd_b_scale',None)
RIXS.mca('SDDB_NOSCALE','Data/sdd_b_mca',None,None)
RIXS.mca('SDDB_N','Data/sdd_b_mca','Data/sdd_b_scale',norm_by = 'Data/i0')
RIXS.mca('XEOL','Data/xeol_a_mca_norm','Data/xeol_a_scale',None)
RIXS.mca('XEOL_N','Data/xeol_a_mca_norm','Data/xeol_a_scale',norm_by = 'Data/i0')
RIXS.mca('XES','Data/mcp_xes_mca','Data/mcp_xes_scale',None)
RIXS.mca('XES_N','Data/mcp_xes_mca','Data/mcp_xes_scale',norm_by = 'Data/i0')

#IMAGE Detectors
RIXS.stack('mcpIMG_A','Data/mcp_a_img',None,None,None)
RIXS.stack('mcpIMG_B','Data/mcp_b_img',None,None,None)

#Log Book for RIXS ES
rixs_log = dict()

rixs_log['Command'] = 'command'
rixs_log['Sample'] = 'Endstation/Sample/Name'
rixs_log['Comments'] = ('comment_01','comment_02','comment_03','comment_04','comment_05','comment_06','comment_07','comment_08','comment_09','comment_10')
rixs_log['Horz (ssh)'] = ['Endstation/Motors/ssh',2]
rixs_log['Vert (ssv)'] = ['Endstation/Motors/ssv',2]
rixs_log['Depth (ssd)'] = ['Endstation/Motors/ssd',2]
rixs_log['Angle (ssa)'] = ['Endstation/Motors/ssa',1]
rixs_log['Temperature'] = ['Endstation/Counters/temp', 1]
rixs_log['Energy'] = ['Beamline/Monochromator/beam',2]
rixs_log['Exit Slit'] = ['Beamline/Apertures/Exit_Slit/vert_gap',1]
rixs_log['Flux'] = 'Beamline/flux'
rixs_log['Dwell'] = ['Endstation/Counters/sec', 1]
rixs_log['Mirror/Grating'] = ('/Beamline/Monochromator/grating','/Beamline/Monochromator/mirror')
rixs_log['Polar/Harmonic'] = ('Beamline/Source/EPU/polarization', 'Beamline/Source/EPU/harmonic')
rixs_log['XES Energy'] = ['Endstation/Detectors/XES/mcp_mca_xes_energy', 2]
rixs_log['XES Grating'] = 'Endstation/Detectors/XES/mcp_mca_xes_grating'
rixs_log['XES Offset'] = ['Endstation/Detectors/XES/mcp_mca_xes_offset', 1]
rixs_log['Shift File'] = 'Endstation/Detectors/XES/mcp_mca_xes_shift_file'
rixs_log['XEOL Rate'] = ['Endstation/Detectors/XEOL/xeol_time_rate_a', 3]
rixs_log['Status'] = 'status'
rixs_log['Date'] = 'date'

