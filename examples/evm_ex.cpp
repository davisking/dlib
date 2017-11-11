
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

const cv::Scalar BLACK( 0, 0, 0 );
const cv::Scalar RED( 255, 0, 0 );
const cv::Scalar GREEN( 0, 255, 0 );
const cv::Scalar BLUE( 0, 0, 255 );
const cv::Scalar WHITE( 255, 255, 255 );
const cv::Scalar ZERO( 0 );


class Tracker : public dlib::correlation_tracker {
public:
  explicit Tracker( unsigned long filter_size = 6,
                    unsigned long num_scale_levels = 5,
                    unsigned long scale_window_size = 23,
                    double regularizer_space = 0.001,
                    double nu_space = 0.025,
                    double regularizer_scale = 0.001,
                    double nu_scale = 0.025,
                    double scale_pyramid_alpha = 1.020
  ) {
    dlib::correlation_tracker( filter_size, num_scale_levels, scale_window_size, regularizer_space, nu_space, regularizer_scale, nu_scale, scale_pyramid_alpha );
    static int instcount = 1;
    m_id = instcount++;
    m_frames_missed = 0;
  }

  int get_id() const {
    return m_id;
  }

  void verify() {
    m_frames_missed = 0;
  }

  int unverify() {
    return ++m_frames_missed;
  }

  int unverified() const {
    return m_frames_missed;
  }

  dlib::drectangle get_refined_position() const {
    return m_refined_position.is_empty() ? get_position() : m_refined_position;
  }

  void set_refined_position( const dlib::drectangle &refined_position ) {
    m_refined_position = refined_position;
  }

private:
  int m_id;
  int m_frames_missed;
  dlib::drectangle m_refined_position;
};



/*
template< typename MatT >
void evm_meanFilter( const MatT &inmat, MatT &outmat, size_t num_iters, int ksize ) {
MatT imat = inmat;
for ( size_t i = 0; i < num_iters; i++ ) {
dlib::gaussian_blur( imat, outmat, 1.0, ksize );
imat.swap( outmat );
}
}

*/

void evm_meanFilter( cv::InputArray _a, cv::OutputArray _b, size_t n = 3, cv::Size s = cv::Size( 5, 5 ) );


void evm_normalization( cv::InputArray iarr, cv::OutputArray oarr ) {
  iarr.getMat().copyTo( oarr );
  cv::Mat omat = oarr.getMat();
  cv::Scalar mean, stdDev;
  cv::meanStdDev( omat, mean, stdDev );
  omat = (omat - mean[0]) / stdDev[0];
}

void evm_meanFilter( cv::InputArray iarr, cv::OutputArray oarr, size_t num_iters, cv::Size ksize ) {
  iarr.getMat().copyTo( oarr );
  cv::Mat omat = oarr.getMat();
  for ( size_t i = 0; i < num_iters; i++ ) {
    cv::blur( omat, omat, ksize );
  }
}

void evm_interpolate_rect( const cv::Rect &arect, const cv::Rect &brect, cv::Rect &orect, double bmix ) {
  double amix = 1 - bmix;
  orect.x = arect.x * amix + brect.x * bmix + 0.5;
  orect.y = arect.y * amix + brect.y * bmix + 0.5;
  orect.width = arect.width * amix + brect.width * bmix + 0.5;
  orect.height = arect.height * amix + brect.height * bmix + 0.5;
}

/**
* Print Mat info such as rows, cols, channels, depth, isContinuous,
* and isSubmatrix.
*/
void cv_dbg_printMatInfo( const std::string& name, cv::InputArray iarr ) {
  cv::Mat imat = iarr.getMat();
  printf( "%s: %dx%d channels=%d depth=%d isContinuous=%s isSubmatrix=%s\n",
          name.c_str(), imat.rows, imat.cols, imat.channels(), imat.depth(),
          (imat.isContinuous() ? "true" : "false"),
          (imat.isSubmatrix() ? "true" : "false") );
}

template<typename DType>
void evm_detrend( cv::InputArray iarr, cv::OutputArray oarr, int lambda = 10 ) {
  CV_DbgAssert( (iarr.type() == CV_32F || iarr.type() == CV_64F)
                && iarr.total() == std::max( iarr.size().width, iarr.size().height ) );

  cv::Mat intermed = iarr.total() == (size_t)iarr.size().height ? iarr.getMat() : iarr.getMat().t();
  if ( intermed.total() < 3 ) {
    intermed.copyTo( oarr );
  } else {
    int total = intermed.total();
    cv::Mat ident = cv::Mat::eye( total, total, intermed.type() );
    cv::Mat d = cv::Mat( cv::Matx<DType, 1, 3>( 1, -2, 1 ) );
    cv::Mat d2Aux = cv::Mat::ones( total - 2, 1, intermed.type() ) * d;
    cv::Mat d2 = cv::Mat::zeros( total - 2, total, intermed.type() );
    for ( int k = 0; k < 3; k++ ) {
      d2Aux.col( k ).copyTo( d2.diag( k ) );
    }
    cv::Mat inter_out = (ident - (ident + lambda * lambda * d2.t() * d2).inv()) * intermed;
    inter_out.copyTo( oarr );
  }
}

template<typename DType>
int countZeros( cv::InputArray iarr ) {
  CV_DbgAssert( iarr.channels() == 1 && iarr.total() == max( iarr.size().width, iarr.size().height ) );

  int count = 0;
  if ( iarr.total() > 0 ) {
    cv::Mat imat = iarr.getMat();
    DType last = imat.at<DType>( 0 );
    for ( int i = 1; i < imat.total(); i++ ) {
      DType current = imat.at<DType>( i );
      if ( (last < 0 && current >= 0) || (last > 0 && current <= 0) ) {
        count++;
      }
      last = current;
    }
  }
  return count;
}

/**
* Same as printMatInfo plus the actual values of the Mat.
* @see printMatInfo
*/
template<typename DType>
void cv_dbg_printMat( const std::string &name, cv::InputArray iarr,
               int rows = -1, int cols = -1, int channels = -1 ) {
  printMatInfo( name, iarr );

  cv::Mat imat = iarr.getMat();
  if ( -1 == rows ) {
    rows = imat.rows;
  }
  if ( -1 == cols ) {
    cols = imat.cols;
  }
  if ( -1 == channels ) {
    channels = imat.channels();
  }

  for ( int y = 0; y < rows; y++ ) {
    cout << "[";
    for ( int x = 0; x < cols; x++ ) {
      DType *e = &imat.at<DType>( y, x );
      cout << "(" << e[0];
      for ( int c = 1; c < channels; c++ ) {
        cout << ", " << e[c];
      }
      cout << ")";
    }
    cout << "]" << endl;
  }
  cout << endl;
}

bool set_video_res( cv::VideoCapture &cap, const std::string &resstr ) {
  if ( resstr.empty() || !cap.isOpened() )
  {
    return false;
  }

  size_t width = cap.get( cv::CAP_PROP_FRAME_WIDTH );
  size_t height = cap.get( cv::CAP_PROP_FRAME_HEIGHT );
  size_t long_hint = std::max( width, height );
  size_t short_hint = std::min( width, height );
  if ( resstr == "low" ) {
    long_hint = 480;
    short_hint = 320;
  } else if ( resstr == "med" || resstr == "medium" ) {
    long_hint = 640;
    short_hint = 480;
  } else if ( resstr == "high" ) {
    long_hint = 1280;
    short_hint = 720;
  } else if ( resstr == "higher" ) {
    long_hint = 1920;
    short_hint = 1280;
  } else if ( resstr == "highest" || resstr == "max" ) {
    long_hint = 0xFFFF;
    short_hint = 0xFFFF;
  } else if ( !resstr.find_first_not_of( "0123456789x" ) == std::string::npos ) {
    std::cout << "bad value for resolution \"" << resstr << "\"." << std::endl;
    return false;
  }
  if ( height > width ) {
    height = long_hint;
    width = short_hint;
  } else {
    width = long_hint;
    height = short_hint;
  }
  if ( resstr.find_first_not_of( "0123456789x" ) == std::string::npos ) {
    size_t xpos = resstr.find_first_of( 'x' );
    width = std::stoul( resstr, &xpos, 10 );
    if ( xpos == resstr.size() ) {
      height = width;
    } else {
      height = std::stoul( resstr.substr( xpos + 1 ), 0, 10 );
    }
  }
  cap.set( cv::CAP_PROP_FRAME_WIDTH, width );
  cap.set( cv::CAP_PROP_FRAME_HEIGHT, height );
  width = cap.get( cv::CAP_PROP_FRAME_WIDTH );
  height = cap.get( cv::CAP_PROP_FRAME_HEIGHT );
  std::cout << "Capturing at " << width << "x" << height << std::endl;

  return true;
}


class EvmGdownIIR {
public:
  EvmGdownIIR();
  virtual ~EvmGdownIIR();

  void onFrame( const cv::Mat &src, cv::Mat &out );

  bool first;
  cv::Size blurredSize;
  double fHigh;
  double fLow;
  int alpha;

private:
  cv::Mat srcFloat;
  cv::Mat blurred;
  cv::Mat lowpassHigh;
  cv::Mat lowpassLow;
  cv::Mat outFloat;
};

EvmGdownIIR::EvmGdownIIR() {
  first = true;
  blurredSize = cv::Size( 10, 10 );
  fLow = 70 / 60.0 / 10;
  fHigh = 80 / 60.0 / 10;
  alpha = 200;
}

EvmGdownIIR::~EvmGdownIIR() {
}

void EvmGdownIIR::onFrame( const cv::Mat &src, cv::Mat &out ) {
  // convert to float
  src.convertTo( srcFloat, CV_32F );

  // apply spatial filter: blur and downsample
  cv::resize( srcFloat, blurred, blurredSize, 0, 0, CV_INTER_AREA );

  if ( first ) {
    first = false;
    blurred.copyTo( lowpassHigh );
    blurred.copyTo( lowpassLow );
    src.copyTo( out );
  } else {
    // apply temporal filter: subtraction of two IIR lowpass filters
    lowpassHigh = lowpassHigh * (1 - fHigh) + fHigh * blurred;
    lowpassLow = lowpassLow * (1 - fLow) + fLow * blurred;
    blurred = lowpassHigh - lowpassLow;

    // amplify
    blurred *= alpha;

    // resize back to original size
    cv::resize( blurred, outFloat, src.size(), 0, 0, CV_INTER_LINEAR );

    // add back to original frame
    outFloat += srcFloat;

    // convert to 8 bit
    outFloat.convertTo( out, CV_8U );
  }
}

class Pulse {
public:
  Pulse();
  virtual ~Pulse();

  void start( int width, int height );
  void onFaceDetection( cv::Mat &frame, int id, const cv::Rect &box, const std::vector<cv::Point> &parts );

  int maxSignalSize;
  double relativeMinFaceSize;
  struct {
    bool enabled;
    int disabledFaceId;
  } faceDetection;
  double fps;
  struct {
    bool magnify;
    double alpha;
  } evm;

  struct Face {
    int id;
    int deleteIn;
    bool selected;

    cv::Rect box;
    std::vector<cv::Point> parts;
    cv::Mat1d timestamps;
    cv::Mat1d raw;
    cv::Mat1d pulse;
    int noPulseIn;
    bool existsPulse;

    cv::Mat1d bpms;
    double bpm;

    struct {
      EvmGdownIIR evm;
      cv::Mat out;
      cv::Rect box;
    } evm;

    struct Peaks {
      cv::Mat1i indices;
      cv::Mat1d timestamps;
      cv::Mat1d values;

      void push( int index, double timestamp, double value );
      void pop();
      void clear();
    } peaks;

    Face();
    Face( int id, const cv::Rect &box, const std::vector<cv::Point> &parts, int deleteIn );

    int nearestBox( const std::vector<cv::Rect> &boxes );
    void updateBox( const cv::Rect &box );
    void reset();
  };

  std::map<int,Face> m_faces;

private:
    int nearestFace( const cv::Rect &box );
    void onFace( cv::Mat &frame, Face &face, const cv::Rect &box );
    void peaks( Face &face );
    void bpm( Face &face );

    double tnow;
    double lastFaceDetectionTimestamp;
    double lastBpmTimestamp;
    cv::Size minFaceSize;
    cv::Mat gray;
    std::vector<cv::Rect> boxes;
    cv::Mat1d powerSpectrum;
    int nextFaceId;
    int deleteFaceIn;
    int holdPulseFor;
    double currentFps;

};

Pulse::Pulse() {
  maxSignalSize = 100;
  relativeMinFaceSize = 0.4;
  deleteFaceIn = 1;
  holdPulseFor = 30;
  fps = 0;
  faceDetection.enabled = true;
  evm.magnify = true;
  evm.alpha = 100;
}

Pulse::~Pulse() {
}

void Pulse::start( int width, int height ) {
  tnow = 0;
  lastFaceDetectionTimestamp = 0;
  lastBpmTimestamp = 0;
  minFaceSize = cv::Size( std::min( width, height ) * relativeMinFaceSize, std::min( width, height ) * relativeMinFaceSize );
  m_faces.clear();
  nextFaceId = 1;
}

void Pulse::onFaceDetection( cv::Mat &frame, int id, const cv::Rect &box, const std::vector<cv::Point> &parts ) {
  // count frames
  tnow = cv::getTickCount();

  auto iter = m_faces.find( id );
  if ( iter != m_faces.end() ) {
    Pulse::Face &face = (*iter).second;
    face.updateBox( box );
    face.parts = parts;
    onFace( frame, face, box );
  } else {
    m_faces[id] = Pulse::Face( id, box, parts, deleteFaceIn );
    onFace( frame, m_faces[id], box );
  }
}

int Pulse::nearestFace( const cv::Rect &box ) {
  int index = -1;
  int min = -1;
  cv::Point pt;

  // search for first unselected face
  for ( auto iter = m_faces.begin(); index < 0 && iter != m_faces.end(); ++iter ) {
    if ( !(*iter).second.selected ) {
      index = (*iter).first;
      pt = box.tl() - (*iter).second.box.tl();
      min = pt.x * pt.x + pt.y * pt.y;
    }
  }

  // no unselected face found
  if ( index == -1 ) {
    return -1;
  }

  // compare with remaining unselected faces
  for ( auto iter = m_faces.find( index ); iter != m_faces.end(); ++iter ) {
    Face &face = (*iter).second;
    if ( !face.selected ) {
      pt = box.tl() - face.box.tl();
      int d = pt.x * pt.x + pt.y * pt.y;
      if ( d < min ) {
        min = d;
        index = (*iter).first;
      }
    }
  }

  return index;
}

void Pulse::onFace( cv::Mat &frame, Face &face, const cv::Rect &box ) {
  // only show magnified face when there is pulse
  cv::Rect evm_box = face.evm.box & cv::Rect( 0,0, frame.cols, frame.rows );
  cv::Mat roi = (!evm.magnify
                 || (evm.magnify && face.existsPulse)
                 || (evm.magnify && !faceDetection.enabled)
                 ? frame( evm_box ) : face.evm.out);

  // if magnification is on
  if ( evm.magnify && evm_box.area() > 1 ) {
    if ( face.evm.evm.first || face.evm.evm.alpha != evm.alpha ) {
      // reset after changing magnify or alpha value
      face.reset();
    }

    // update face Eulerian video magnification values
    face.evm.evm.alpha = evm.alpha;

    // apply Eulerian video magnification on face box
    try {
      face.evm.evm.onFrame( frame( evm_box ), roi );
    } catch ( std::exception &ex ) {
      std::cout << "bad roi " << evm_box << "frame size:" << frame.cols << "x" << frame.rows << std::endl;
    }

  } else if ( !face.evm.evm.first ) {
    // reset after changing magnify value
    face.reset();
  }

  // capture raw value and timestamp, shifting if necessary
  if ( face.raw.rows >= maxSignalSize ) {
    // shift raw and timestamp
    const int total = face.raw.rows;
    face.raw.rowRange( 1, total ).copyTo( face.raw.rowRange( 0, total - 1 ) );
    face.raw.pop_back();
    face.timestamps.rowRange( 1, total ).copyTo( face.timestamps.rowRange( 0, total - 1 ) );
    face.timestamps.pop_back();
  }
  // push back raw and timestamp
  face.raw.push_back<double>( mean( roi )(1) ); // grab green channel
  face.timestamps.push_back<double>( cv::getTickCount() );

  // verify if raw signal is stable enough
  cv::Scalar rawStdDev;
  cv::meanStdDev( face.raw, cv::Scalar(), rawStdDev );
  const bool stable = rawStdDev( 0 ) <= (evm.magnify ? 1 : 0) * evm.alpha * 0.045 + 1;
  
  if ( stable ) {
    // calculate current FPS
    currentFps = this->fps;
    if ( currentFps == 0 ) {
      const double diff = (face.timestamps( face.timestamps.rows - 1 ) - face.timestamps( 0 )) * 1000.0 / cv::getTickFrequency();
      currentFps = face.timestamps.rows * 1000.0 / diff;
    }

    // process raw signal
    evm_detrend<double>( face.raw, face.pulse, 0.5 * currentFps );
    evm_normalization( face.pulse, face.pulse );
    evm_meanFilter( face.pulse, face.pulse );

    // detects peaks and validates pulse
    peaks( face );
  } else {
    // too noisy to detect pulse
    face.existsPulse = false;

    // reset to improve BPM detection speed
    if ( faceDetection.enabled ) {
      face.reset();
    }
  }

  if ( face.existsPulse ) {
    bpm( face );
  }

  if ( !face.existsPulse ) {
    if ( face.pulse.rows == face.raw.rows ) {
      face.pulse = 0;
    } else {
      face.pulse = cv::Mat1d::zeros( face.raw.rows, 1 );
    }
    face.peaks.clear();
    face.bpms.pop_back( face.bpms.rows );
    face.bpm = 0;
  }

#ifndef __ANDROID__
 // draw( frame, face, box );
#endif
}

// Algorithm based on: Pulse onset detection
void Pulse::peaks( Face& face ) {
  // remove old peaks
  face.peaks.clear();

  int lastIndex = 0;
  int lastPeakIndex = 0;
  int lastPeakTimestamp = face.timestamps( 0 );
  int lastPeakValue = face.pulse( 0 );
  double peakValueThreshold = 0;

  for ( int i = 1; i < face.raw.rows; i++ ) {
    // if more than X milliseconds passed search for peak in this segment
    const double diff = (face.timestamps( i ) - face.timestamps( lastIndex )) * 1000.0 / cv::getTickFrequency();
    if ( diff >= 200 ) {
      // find max in this segment
      int relativePeakIndex[2];
      double peakValue;
      cv::minMaxIdx( face.pulse.rowRange( lastIndex, i + 1 ), 0, &peakValue, 0, &relativePeakIndex[0] );
      const int peakIndex = lastIndex + relativePeakIndex[0];

      // if max is not at the boundaries and is higher than the threshold
      if ( peakValue > peakValueThreshold && lastIndex < peakIndex && peakIndex < i ) {
        const double peakTimestamp = face.timestamps( peakIndex );
        const double peakDiff = (peakTimestamp - lastPeakTimestamp) * 1000.0 / cv::getTickFrequency();

        // if peak is too close to last peak and has an higher value
        if ( peakDiff <= 200 && peakValue > lastPeakValue ) {
          // then pop last peak
          face.peaks.pop();
        }

        // if peak is far away enough from last peak or has an higher value
        if ( peakDiff > 200 || peakValue > lastPeakValue ) {
          // then store current peak
          face.peaks.push( peakIndex, peakTimestamp, peakValue );

          lastPeakIndex = peakIndex;
          lastPeakTimestamp = peakTimestamp;
          lastPeakValue = peakValue;

          peakValueThreshold = 0.6 * mean( face.peaks.values )(0);
        }
      }

      lastIndex = i;
    }
  }

  // verify if peaks describe a valid pulse signal
  cv::Scalar peakValuesStdDev;
  cv::meanStdDev( face.peaks.values, cv::Scalar(), peakValuesStdDev );
  const double diff = (face.timestamps( face.raw.rows - 1 ) - face.timestamps( 0 )) / cv::getTickFrequency();

  cv::Scalar peakTimestampsStdDev;
  if ( face.peaks.indices.rows >= 3 ) {
    cv::meanStdDev( (face.peaks.timestamps.rowRange( 1, face.peaks.timestamps.rows ) -
                     face.peaks.timestamps.rowRange( 0, face.peaks.timestamps.rows - 1 )) /
                      cv::getTickFrequency(), cv::Scalar(), peakTimestampsStdDev );
  }

  // TODO extract constants to class?
  bool validPulse = (2 <= face.peaks.indices.rows &&
                     40 / 60 * diff <= face.peaks.indices.rows &&
                     face.peaks.indices.rows <= 240 / 60 * diff &&
                     peakValuesStdDev( 0 ) <= 0.5 &&
                     peakTimestampsStdDev( 0 ) <= 0.5);

  if ( !face.existsPulse && validPulse ) {
    // pulse become valid
    face.noPulseIn = holdPulseFor;
    face.existsPulse = true;
  } else if ( face.existsPulse && !validPulse ) {
    // pulse become invalid
    if ( face.noPulseIn > 0 ) {
      face.noPulseIn--; // keep pulse for a few frames
    } else {
      face.existsPulse = false; // pulse has been invalid for too long
    }
  }
}

void Pulse::Face::Peaks::push( int index, double timestamp, double value ) {
  indices.push_back<int>( index );
  timestamps.push_back<double>( timestamp );
  values.push_back<double>( value );
}

void Pulse::Face::Peaks::pop() {
  indices.pop_back( std::min( indices.rows, 1 ) );
  timestamps.pop_back( std::min( timestamps.rows, 1 ) );
  values.pop_back( std::min( values.rows, 1 ) );
}

void Pulse::Face::Peaks::clear() {
  indices.pop_back( indices.rows );
  timestamps.pop_back( timestamps.rows );
  values.pop_back( values.rows );
}

void Pulse::bpm( Face& face ) {
  cv::dft( face.pulse, powerSpectrum );

  const int total = face.raw.rows;

  // band limit
  const int low = total * 40.0 / 60.0 / currentFps + 1;
  const int high = total * 240.0 / 60.0 / currentFps + 1;
  powerSpectrum.rowRange( 0, std::min( (size_t)low, (size_t)total ) ) = ZERO;
  powerSpectrum.pop_back( std::min( (size_t)(total - high), (size_t)total ) );

  // power spectrum
  pow( powerSpectrum, 2, powerSpectrum );

  if ( !powerSpectrum.empty() ) {
    // grab index of max power spectrum
    int idx[2] = { 0 };
    cv::minMaxIdx( powerSpectrum, 0, 0, 0, &idx[0] );

    // calculate BPM
    face.bpms.push_back<double>( idx[0] * currentFps * 30.0 / total ); // constant 30 = 60 BPM / 2
  }

  // update BPM when none available or after one second
  if ( face.bpm == 0 || (tnow - lastBpmTimestamp) * 1000.0 / cv::getTickFrequency() >= 1000 ) {
    lastBpmTimestamp = cv::getTickCount();

    // average calculated BPMs since last time
    face.bpm = mean( face.bpms )(0);
    face.bpms.pop_back( face.bpms.rows );

    // mark as no pulse when BPM is too low
    if ( face.bpm <= 40 ) {
      face.existsPulse = false;
    }
  }
}

void pulse_draw_face( dlib::image_window &dsp, const Pulse::Face &face, const cv::Rect &box ) {
  const dlib::rgb_pixel RED = dlib::rgb_pixel( 255, 0, 0 );
  const dlib::rgb_pixel GREEN = dlib::rgb_pixel( 0, 0, 255 );
  const dlib::rgb_pixel BLUE = dlib::rgb_pixel( 0, 0, 255 );
  const dlib::rgb_pixel BROWN = dlib::rgb_pixel( 155, 0, 100 );

  std::vector<dlib::image_display::overlay_rect> rects;
  rects.push_back( dlib::image_display::overlay_rect( dlib::rectangle( box.x, box.y, box.x + box.width, box.y + box.height ), BLUE ) );
  //rects.push_back( dlib::image_display::overlay_rect( dlib::rectangle( face.box.x, face.box.y, face.box.x + face.box.width, face.box.y + face.box.height ), BLUE ) );
  //rects.push_back( dlib::image_display::overlay_rect( dlib::rectangle( face.evm.box.x, face.evm.box.y, face.evm.box.x + face.evm.box.width, face.evm.box.y + face.evm.box.height ), GREEN ) );

  // bottom left point of face box
  dlib::point tr( box.x, box.y );
  dlib::point bl( box.x, box.y + box.height );
  dlib::point lastpt = bl;
  dlib::point endpt;
  std::vector<dlib::image_display::overlay_line> lines;
  std::vector<dlib::image_display::overlay_line> points;
  for ( int i = 0; i < face.raw.rows; i++ ) {
    endpt = bl + dlib::point( i, -face.raw( i ) + 50 );
    lines.push_back( dlib::image_display::overlay_line( lastpt, endpt, GREEN ) );
    lastpt = endpt;
    endpt = bl + dlib::point( i, -face.pulse( i ) * 10 - 50 );
    lines.push_back( dlib::image_display::overlay_line( endpt, endpt, (face.existsPulse ? RED : BROWN) ) );
  }

  // peaks
  std::vector<dlib::image_display::overlay_circle> circles;
  for ( int i = 0; i < face.peaks.indices.rows; i++ ) {
    const int index = face.peaks.indices( i );
    endpt = bl + dlib::point( index, -face.pulse( index ) * 10 - 50 );
    circles.push_back( dlib::image_display::overlay_circle( endpt, 2, BLUE ) );
  }

  dsp.add_overlay( rects );
  dsp.add_overlay( lines );
  dsp.add_overlay( points );
  dsp.add_overlay( circles );

  // id
  dsp.add_overlay( dlib::image_display::overlay_rect( dlib::rectangle( tr, tr ), BLUE, cv::format( "%d", face.id ) ) );

  // bpm
  dsp.add_overlay( dlib::image_display::overlay_rect( dlib::rectangle( bl, bl ), RED, cv::format( "%.3f", face.bpm ) ) );

}

Pulse::Face::Face() {
  this->id = -1;
  this->deleteIn = 0;
  this->box = cv::Rect( 0, 0, 1, 1 );
  this->updateBox( this->box );
  this->existsPulse = false;
  this->noPulseIn = 0;
}

Pulse::Face::Face( int id, const cv::Rect &box, const std::vector<cv::Point> &parts, int deleteIn ) {
  this->id = id;
  this->parts = parts;
  this->box = box;
  this->deleteIn = deleteIn;
  this->updateBox( this->box );
  this->existsPulse = false;
  this->noPulseIn = 0;
}

int Pulse::Face::nearestBox( const std::vector<cv::Rect>& boxes ) {
  if ( boxes.empty() ) {
    return -1;
  }
  int index = 0;
  cv::Point p = box.tl() - boxes.at( 0 ).tl();
  int min = p.x * p.x + p.y * p.y;
  for ( size_t i = 1; i < boxes.size(); i++ ) {
    p = box.tl() - boxes.at( i ).tl();
    int d = p.x * p.x + p.y * p.y;
    if ( d < min ) {
      min = d;
      index = i;
    }
  }
  return index;
}

void Pulse::Face::updateBox( const cv::Rect& a ) {
  // update box position and size
  cv::Point p = box.tl() - a.tl();
  double d = (p.x * p.x + p.y * p.y) / pow( box.width / 3., 2. );
  evm_interpolate_rect( box, a, box, std::min( 1., d ) );

  // update EVM box
  cv::Point c = box.tl() + cv::Point( box.size().width * .5, box.size().height * .5 );
  cv::Point r( box.width * .275, box.height * .425 );
  evm.box = cv::Rect( c - r, c + r );
  //std::cout << "evm_box:" << evm.box << " orig:" << a << " interp:" << box << std::endl;
}

void Pulse::Face::reset() {
  // restarts Eulerian video magnification
  evm.evm.first = true;

  // clear raw signal
  raw.pop_back( raw.rows );
  timestamps.pop_back( timestamps.rows );
}

std::vector<cv::Point3d> get_3d_model_points() {
  std::vector<cv::Point3d> modelPoints;

  modelPoints.push_back( cv::Point3d( 0.0f, 0.0f, 0.0f ) ); //The first must be (0,0,0) while using POSIT
  modelPoints.push_back( cv::Point3d( 0.0f, -330.0f, -65.0f ) );
  modelPoints.push_back( cv::Point3d( -225.0f, 170.0f, -135.0f ) );
  modelPoints.push_back( cv::Point3d( 225.0f, 170.0f, -135.0f ) );
  modelPoints.push_back( cv::Point3d( -150.0f, -150.0f, -125.0f ) );
  modelPoints.push_back( cv::Point3d( 150.0f, -150.0f, -125.0f ) );

  return modelPoints;

}

std::vector<cv::Point2d> get_2d_image_points( const dlib::full_object_detection &det ) {
  std::vector<cv::Point2d> image_points;
  image_points.push_back( cv::Point2d( det.part( 30 ).x(), det.part( 30 ).y() ) );    // Nose tip
  image_points.push_back( cv::Point2d( det.part( 8 ).x(), det.part( 8 ).y() ) );      // Chin
  image_points.push_back( cv::Point2d( det.part( 36 ).x(), det.part( 36 ).y() ) );    // Left eye left corner
  image_points.push_back( cv::Point2d( det.part( 45 ).x(), det.part( 45 ).y() ) );    // Right eye right corner
  image_points.push_back( cv::Point2d( det.part( 48 ).x(), det.part( 48 ).y() ) );    // Left Mouth corner
  image_points.push_back( cv::Point2d( det.part( 54 ).x(), det.part( 54 ).y() ) );    // Right mouth corner
  return image_points;
}

cv::Mat get_camera_matrix( float focal_length, cv::Point2d center ) {
  cv::Mat camera_matrix = (cv::Mat_<double>( 3, 3 ) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
  return camera_matrix;
}

void match_faces_to_trackers( const dlib::cv_image<dlib::bgr_pixel> &cimg, const std::vector<dlib::rectangle> &faces, std::vector<Tracker> &trackers ) {
  std::vector<dlib::rectangle> remaining_faces = faces;

  auto titer = trackers.begin();
  while ( titer != trackers.end() ) {
    Tracker &tracker = *titer;
    dlib::drectangle trect = tracker.get_refined_position();
    dlib::dpoint tcenter = dlib::center( trect );
    double distsq = trect.width() * trect.width() * 0.125;
    bool is_match = false;
    auto fiter = remaining_faces.begin();
    while ( !is_match && fiter != remaining_faces.end() ) {
      dlib::rectangle &face_rect = *fiter;
      is_match = (tcenter - dlib::center( face_rect )).length_squared() < distsq;
      if ( is_match ) {
        tracker.set_refined_position( face_rect );
        tracker.start_track( cimg, face_rect );
        fiter = remaining_faces.erase( fiter );
      } else {
        fiter++;
      }
    }
    if ( !is_match ) {
      int misses = tracker.unverify();
      if ( misses > 3 ) {
        titer = trackers.erase( titer );
      } else {
        titer++;
      }
    } else {
      tracker.verify();
      titer++;
    }
  }
  for ( const auto &face_rect : remaining_faces ) {
    Tracker tracker;
    tracker.start_track( cimg, face_rect );
    tracker.set_refined_position( face_rect );
    trackers.push_back( tracker );
  }
}


class face_window : public dlib::image_window {
public:
  face_window() : dlib::image_window() {}

  template < typename image_type >
  face_window(
    const image_type& img,
    const std::string& title
  ) : dlib::image_window( img, title ) {
  }


  ~face_window(
  ) {
  }


  void get_last_key(
    unsigned long &key,
    bool &is_printable,
    unsigned long &state
  ) {
    key = last_key;
    is_printable = last_key_printable;
    state = last_key_state;
    last_key = 0;
    last_key_printable = false;
    last_key_state = 0;
  }

private:

  unsigned long last_key = 0;
  unsigned long last_key_state = 0;
  bool last_key_printable = false;

  virtual void on_keydown(
    unsigned long key,
    bool is_printable,
    unsigned long state
  ) {
    //dlib::image_window::on_keydown( key, is_printable, state );
    last_key = key;
    last_key_printable = is_printable;
    last_key_state = state;
  }

  // restricted functions
  face_window( face_window& ) {}
  face_window& operator= ( face_window& ) {}
};

// ----------------------------------------------------------------------------------------

int main( int argc, char *argv[] ) {
  try {
    cv::VideoCapture cap( 1 );
    if ( !cap.isOpened() ) {
      cap.open( 0 );
    }
    if ( !cap.isOpened() ) {
      std::cerr << "Unable to connect to camera" << std::endl;
      return 1;
    }
    std::vector<std::string> resolutions = { "low", "medium", "high", "higher", "max" };
    int videores = 2;

    face_window win;
    double fps = 30.0; // Just a place holder. Actual value calculated after 100 frames.
    int64 tnow = cv::getTickCount();
    int64 last_time = tnow;
    int64 last_dt = 3;
    int64 total_dt = 300;
    int64 last_detect = 0;

    // Load face detection and pose estimation models.
    std::vector<Tracker> trackers;
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor pose_model;
    dlib::deserialize( "shape_predictor_68_face_landmarks.dat" ) >> pose_model;
    Pulse pulse;

    // Grab and process frames until the main window is closed by the user.
    while ( !win.is_closed() ) {
      // Grab a frame
      cv::Mat temp;
      cap >> temp;
      // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
      // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
      // long as temp is valid.  Also don't do anything to temp that would cause it
      // to reallocate the memory which stores the image as that will make cimg
      // contain dangling pointers.  This basically means you shouldn't modify temp
      // while using cimg.
      dlib::cv_image<dlib::bgr_pixel> cimg( temp );
      std::vector<dlib::image_display::overlay_rect> rects;

      tnow = cv::getTickCount();

      for ( Tracker &tracker : trackers ) {
        tracker.update( cimg, tracker.get_refined_position() );
        tracker.set_refined_position( tracker.get_position() );
      }
      if ( (tnow - last_detect) / cv::getTickFrequency() > 0.25 ) {
        std::vector<dlib::rectangle> faces = detector( cimg );
        match_faces_to_trackers( cimg, faces, trackers );
        last_detect = tnow;
      }
      // Find the pose of each face.
      dlib::rectangle img_area = dlib::get_rect( cimg );
      std::vector<dlib::full_object_detection> shapes;

      for ( Tracker &tracker : trackers ) {
        dlib::drectangle fr = tracker.get_refined_position();
        dlib::full_object_detection shape = pose_model( cimg, fr );
        dlib::dpoint eye_center = (shape.part( 36 ) + shape.part( 39 ) + shape.part( 42 ) + shape.part( 45 )) * 0.25;
        dlib::dpoint nose_bottom = (shape.part( 30 ) + shape.part( 32 ) + shape.part( 33 ) + shape.part( 34 )) * 0.25;
        double head_size = 3.5*dlib::length( eye_center - nose_bottom );
        dlib::drectangle face_center = dlib::centered_drect( eye_center, 0.5 * (head_size + fr.width()), 0.5 *(head_size + fr.height()) );
        tracker.set_refined_position( face_center );
        std::vector<cv::Point> parts;
        for ( int part_i = 0; part_i < shape.num_parts(); part_i++ ) {
          parts.push_back( cv::Point( shape.part( part_i ).x(), shape.part( part_i ).y() ) );
        }
        dlib::chip_details chip = dlib::get_face_chip_details( shape );
        dlib::array2d<dlib::bgr_pixel> chipimg;
        dlib::extract_image_chip( cimg, chip, chipimg );
        cv::Mat cvchipimg = dlib::toMat( chipimg );
        dlib::point_transform_affine xfrm = dlib::get_mapping_to_chip( chip );
        dlib::dpoint chip_eye_center = xfrm( eye_center );
        dlib::dpoint chip_nose_bottom = xfrm( nose_bottom );
        double chip_head_size = 2*std::max( dlib::length( chip_eye_center - chip_nose_bottom ), 2.0 );
        dlib::drectangle chip_face_center = dlib::centered_drect( chip_eye_center, chip_head_size, chip_head_size );

        if ( chip_face_center.left() > chipimg.nc() - 2.0 ) {
          chip_face_center.left() = chipimg.nc() - 2.0;
        }
        if ( chip_face_center.bottom() < 2.0 ) {
          chip_face_center.bottom() = 2.0;
        }
        if ( chip_face_center.top() > chipimg.nr() - 2.0 ) {
          chip_face_center.top() = chipimg.nr() - 2.0;
        }
        if ( chip_face_center.right() < 2.0 ) {
          chip_face_center.right() = 2.0;
        }
        //std::cout << "eye_center: " << chip_eye_center << " dist: " << chip_head_size << " face center: " << face_center << std::endl;

        chip_face_center = chip_face_center.intersect( dlib::drectangle( 1, 1,chipimg.nc()-1, chipimg.nr()-1 ) );
        //std::cout << "after isect: " << face_center << std::endl;
        cv::Rect evm_roi( chip_face_center.right(), chip_face_center.top(), chip_face_center.width(), chip_face_center.height() );
        //cv::Rect evm_roi( 0, 0, cvchipimg.cols, cvchipimg.rows );
        pulse.onFaceDetection( cvchipimg, tracker.get_id(), evm_roi, parts );
        shapes.push_back( shape );
        rects.push_back( dlib::image_display::overlay_rect( chip_face_center, dlib::rgb_pixel( 100, 100, 100 ), cv::format( "%d m %d", tracker.get_id(), tracker.unverified() ) ) );
      }

      std::vector<dlib::image_display::overlay_line> lines = dlib::render_face_detections( shapes );
      for ( const dlib::full_object_detection &shape : shapes ) {
        //lines.push_back( face_pose_line( cimg, shape, rgb_pixel( 0, 0, 255 ) ) );
      }

      tnow = cv::getTickCount();
      int64 dt = tnow - last_time;
      total_dt = (total_dt + dt) - last_dt;
      fps = ((double)cv::getTickFrequency()) / total_dt;
      last_time = tnow;
      last_dt = dt;

      // Display it all on the screen
      win.clear_overlay();
      win.set_image( cimg );

      win.add_overlay( lines );
      win.add_overlay( rects );
      for ( const auto &tracker : trackers ) {
        dlib::drectangle dbox = tracker.get_refined_position();
        cv::Rect box( dbox.left(), dbox.top(), dbox.width(), dbox.height() );
        pulse_draw_face( win, pulse.m_faces[tracker.get_id()], box );
      }
      win.add_overlay( dlib::rectangle( 50, cimg.nr() - 50, 50, cimg.nr() - 50 ),
                       dlib::rgb_pixel( 255, 255, 255 ), cv::format( "FPS %.2f", fps ) );

      unsigned long key = 0;
      bool key_is_print = false;
      unsigned long key_state = 0;
      win.get_last_key( key, key_is_print, key_state );
      switch ( key ) {
      case dlib::base_window::KEY_ESC: case 'q':
        exit( 0 );
        break;
      case 's': // snap an image
        //snap( tnow, cimg, shapes );
        break;
      case 'v':
        videores = (videores + 1) % resolutions.size();
        set_video_res( cap, resolutions[videores] );
        break;
      default:
        break;
      }
    }
  } catch ( dlib::serialization_error &ex ) {
    std::cout << "You need dlib's default face landmarking model file to run this example." << std::endl;
    std::cout << "You can get it from the following URL: " << std::endl;
    std::cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << std::endl;
    std::cout << std::endl << ex.what() << std::endl;
  } catch ( std::exception &ex ) {
    std::cout << ex.what() << std::endl;
  }
}

void chips( const std::vector<dlib::full_object_detection> &shapes ) {
  std::vector<dlib::chip_details> chips = dlib::get_face_chip_details( shapes );

  for ( int i = 0; i < chips.size(); i++ ) {
    dlib::point_transform_affine xfrm = dlib::get_mapping_to_chip( chips[i] );
    dlib::point point_in_chip = xfrm( shapes[i].part( 0 ) );
  }
}
