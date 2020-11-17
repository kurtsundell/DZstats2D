function varargout = FAQs(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',mfilename,'gui_Singleton',gui_Singleton,'gui_OpeningFcn',@FAQs_OpeningFcn,'gui_OutputFcn',@FAQs_OutputFcn,'gui_LayoutFcn',[],'gui_Callback',[]);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end
if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

function FAQs_OpeningFcn(hObject, eventdata, H, varargin)
H.output = hObject;
guidata(hObject, H);
function varargout = FAQs_OutputFcn(hObject, eventdata, H) 
varargout{1} = H.output;

function listbox3_Callback(hObject, eventdata, H)
