from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QToolTip, QLabel, QVBoxLayout, QSlider, QGridLayout
from qtpy.QtGui import QFont, QPixmap, QImage, QCursor
from qtpy.QtCore import Qt, QTimer
import qtpy.QtCore as QtCore
import numpy as np
import sys
import cv2

class mainView(QMainWindow):

	def __init__(self, stack, positions = False, label = None):
		super().__init__()
		self.thresh = False
		self.positions = positions
		self.label = label
		# convert 16 bit grayscale to 8 bit
		# by mapping the data range to 0 - 255
		if stack.dtype == 'uint16':
			self.stack = ((stack - stack.min()) / (stack.ptp() / 255.0)).astype(np.uint8) 
		else:
			self.stack = stack
		self.stack_length = self.stack.shape[0]

		self.initUI()


	def initUI(self):
		#initialise UI
		self.setWindowTitle('CTFishPy')
		self.statusBar().showMessage('Status bar: Ready')

		self.viewer = Viewer(self.stack, label = self.label, positions = self.positions, parent=self)
		self.setCentralWidget(self.viewer)
		#widget.findChildren(QWidget)[0]

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		self.setGeometry(1100, 10, self.viewer.width(), self.viewer.height())
		
	def keyPressEvent(self, event):
		#close window if esc or q is pressed
		if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q :
			self.close()

	def updateStatusBar(self, slice_ = 0):
		self.statusBar().showMessage(f'{slice_}/{self.stack_length}')


class Viewer(QWidget):

	def __init__(self, stack, label = None, positions = None, parent=None):
		super().__init__()

		# init variables
		if np.max(stack) == 1: stack = stack*255 #fix labels so you can view them if are main stack
		self.ogstack = stack
		self.stack_size = stack.shape[0]
		self.slice = 0
		self.min_thresh = 0
		self.max_thresh = 150
		self.thresh = False # turn off for now
		self.label = label
		self.pad = 20
		self.parent = parent
		# unpack images to convert each one to grayscale
		self.ogstack = np.array([np.stack((img,)*3, axis=-1) for img in self.ogstack])

		self.scale = 300
		#resize image
		im = self.ogstack[0]
		width = int(im.shape[1] * self.scale / 100)
		height = int(im.shape[0] * self.scale / 100)
		dim = (width, height)
		# resize image
		resized = [cv2.resize(img, dim, interpolation = cv2.INTER_AREA) for img in self.ogstack]
		self.ogstack = np.array(resized)
		scale_factor = self.scale / 100
		self.positions = [[z, y*scale_factor, x*scale_factor] for z, y, x in positions]

		# label stack
		if self.label is not None:
			self.thresh == False

			# change pixels to red if in label
			self.ogstack[label == 255, :] = [255, 69, 0]

		if self.positions is not None:
			for z, y, x in self.positions:
				z, y, x = int(z), int(y), int(x)
				cv2.rectangle(self.ogstack[z], (x - 2, y - 2), (x + 2, y + 2), (255,69,0), -1)
				cv2.circle(self.ogstack[z], (x, y), 10, (0, 255, 0), 2)
		#if self.ogstack.shape[0] == self.ogstack.shape[1]: self.is_single_image = True
		if len(self.ogstack.shape) == 2: self.is_single_image = True
		else: self.is_single_image = False


		# set background colour to cyan
		p = self.palette()
		p.setColor(self.backgroundRole(), Qt.cyan)
		self.setPalette(p)
		self.setAutoFillBackground(True)

		self.qlabel = QLabel(self)
		self.qlabel.setMargin(10)
		self.initUI()

	def initUI(self):
		pad = self.pad
		self.update()
		if self.is_single_image == False:
			self.initSlider()
			self.slider.valueChanged.connect(self.updateSlider)
			self.setGeometry(0, 0, 
			self.pixmap.width() + pad, 
			self.pixmap.height() + pad + self.slider.height()*2 + pad)

		elif self.is_single_image: 
			self.setGeometry(0, 0, 
			self.pixmap.width() + pad, 
			self.pixmap.height() + pad*3)

		if self.thresh == True: 
			self.initThresholdSliders()
			self.setGeometry(0, 0, 
				self.pixmap.width() + pad, 
				self.pixmap.height()+ pad + self.slider.height()*2 + self.min_thresh_slider.height()*2 + self.max_thresh_slider.height()*2)
			self.min_thresh_slider.valueChanged.connect(self.update_min_thresh)
			self.max_thresh_slider.valueChanged.connect(self.update_max_thresh)

		
			
		

	def update(self):

		if self.slice > self.stack_size-1: 	self.slice = 0
		if self.slice < 0: 					self.slice = self.stack_size-1
		
		elif self.is_single_image == True: self.image = self.ogstack
		else: self.image = self.ogstack[self.slice]

		# transform image to qimage and set pixmap
		self.image = self.np2qt(self.image)
		self.pixmap = QPixmap(QPixmap.fromImage(self.image))
		self.qlabel.setPixmap(self.pixmap)

	def wheelEvent(self, event):
		if self.is_single_image: return
		#scroll through slices and go to beginning if reached max
		self.slice = self.slice + int(event.angleDelta().y()/120)
		if self.slice > self.stack_size-1: 	self.slice = 0
		if self.slice < 0: 					self.slice = self.stack_size-1
		self.slider.setValue(self.slice)
		self.update()

	def np2qt(self, image):
		# transform np cv2 image to qt format

		# check length of image shape to check if image is grayscale or color
		if len(image.shape) == 2: grayscale = True
		elif len(image.shape) == 3: grayscale = False
		else: raise ValueError('[Viewer] Cant tell if stack is color or grayscale, weird shape :/')
		if self.is_single_image: grayscale = True

		# convert npimage to qimage depending on color mode
		if grayscale == True:
			height, width = self.image.shape
			bytesPerLine = width
			return QImage(image.data, width, height, bytesPerLine, QImage.Format_Indexed8)
		else:
			height, width, channel = image.shape
			bytesPerLine = 3 * width
			return QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)

	def initSlider(self):
		self.slider = QSlider(Qt.Horizontal, self)
		self.slider.setMinimum(0)
		self.slider.setMaximum(self.stack_size)
		self.slider.setGeometry(10, self.pixmap.height()+10, self.pixmap.width(), 20)

	def updateSlider(self):
		self.slice = self.slider.value()
		self.update()
		self.parent.updateStatusBar(self.slice)

	def initThresholdSliders(self):
		self.min_thresh_slider = QSlider(Qt.Horizontal, self)
		self.min_thresh_slider.setMinimum(0) # betweeen 100 and 200
		self.min_thresh_slider.setMaximum(150) # -for decimal places
		self.min_thresh_slider.setSingleStep(1)
		self.min_thresh_slider.setGeometry(10, self.pixmap.height()+10+self.slider.height(), self.pixmap.width(), 20)
		
		self.max_thresh_slider = QSlider(Qt.Horizontal, self)
		self.max_thresh_slider.setMinimum(100)
		self.max_thresh_slider.setMaximum(255)
		self.max_thresh_slider.setSingleStep(1)
		self.max_thresh_slider.setGeometry(10, self.pixmap.height()+10+self.slider.height()+self.min_thresh_slider.height(), self.pixmap.width(), 20)

	def update_min_thresh(self):
		self.min_thresh = self.min_thresh_slider.value() # divide by 100 to get decimal points
		self.update()

	def update_max_thresh(self):
		self.max_thresh = self.max_thresh_slider.value() # divide by 100 to get decimal points
		self.update()


def mainViewer(stack, positions=None, labels=None):
	app = QApplication(sys.argv)
	win = mainView(stack, positions, labels)
	win.show()
	app.exec_()
	return

'''
				listOfCoordinates = list(zip(indices[0], indices[1]))
				if listOfCoordinates != []:
					for (x, y) in listOfCoordinates:
						print(x, y)
									megastack = zip(self.ogstack, self.label)
			labelled_stack = []
			for (img, l) in megastack:
				#indices = np.where(l == 1)
				img[l == 1, 0] = 255

				labelled_stack.append(img)
			self.ogstack = np.array(labelled_stack)
'''