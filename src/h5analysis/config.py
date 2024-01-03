import parse

class h5Config:
    def __init__(self):
        self.h5dict = dict()
        self.sca_folders = list()
        
    def key(self,pattern,scan_var):
        self.h5key = pattern
        self.scan_var = scan_var
        
    def get_h5key(self,scan):
        kwargs = dict()
        kwargs[self.scan_var] = scan
        return self.h5key.format(**kwargs)
    
    def get_h5scan(self,key):
        return parse.parse(self.h5key,key)[self.scan_var]
    
    def get_path(self,scan,path):
        if self.h5key == '':
            return path
        else:
            return f"{self.get_h5key(scan)}/{path}"
        
    def sca_folder(self,path):
        self.sca_folders.append(path)
        
    def sca(self,alias,path,norm_by=None):
        dataDict = dict()
        dataDict['type'] = 'SCA'
        dataDict['SCA_Path'] = path
        dataDict['norm_by'] = norm_by
        self.h5dict[alias] = dataDict
        
    def mca(self,alias,path,scale,norm_by=None):
        dataDict = dict()
        dataDict['type'] = 'MCA'
        dataDict['MCA_Path'] = path
        dataDict['MCA_Scale'] = scale
        dataDict['norm_by'] = norm_by
        self.h5dict[alias] = dataDict
        
    def stack(self,alias,path,scale1,scale2,norm_by=None):
        dataDict = dict()
        dataDict['type'] = 'STACK'
        dataDict['STACK_Path'] = path
        dataDict['MCA_Scale'] = scale1
        dataDict['STACK_Scale'] = scale2
        dataDict['norm_by'] = norm_by
        self.h5dict[alias] = dataDict