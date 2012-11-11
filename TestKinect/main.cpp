#include "libfreenect.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "Mutex.h"
#include "MyFreenectDevice.h"
#include <list>
#define THRESH 2
#define THRESH_CURVATURE 200
#define THRESH_MASK 10
#define THRESH_PORCENT 0.05

using namespace cv;
using namespace std;
/*
class Mutex {
public:
	Mutex() {
		pthread_mutex_init( &m_mutex, NULL );
	}
	void lock() {
		pthread_mutex_lock( &m_mutex );
	}
	void unlock() {
		pthread_mutex_unlock( &m_mutex );
	}
private:
	pthread_mutex_t m_mutex;
};
*/
class MyFreenectDevice : public Freenect::FreenectDevice {
  public:
	MyFreenectDevice(freenect_context *_ctx, int _index)
		: Freenect::FreenectDevice(_ctx, _index), m_buffer_depth(FREENECT_DEPTH_11BIT),m_buffer_rgb(FREENECT_VIDEO_RGB), m_gamma(2048), m_new_rgb_frame(false), m_new_depth_frame(false),
		  depthMat(Size(640,480),CV_16UC1), rgbMat(Size(640,480),CV_8UC3,Scalar(0)), ownMat(Size(640,480),CV_8UC3,Scalar(0))
	{
		for( unsigned int i = 0 ; i < 2048 ; i++) {
			float v = i/2048.0;
			v = std::pow(v, 3)* 6;
			m_gamma[i] = v*6*256;
		}
	}
	// Do not call directly even in child
	void VideoCallback(void* _rgb, uint32_t timestamp) {
		m_rgb_mutex.lock();
		uint8_t* rgb = static_cast<uint8_t*>(_rgb);
		rgbMat.data = rgb;
		m_new_rgb_frame = true;
		m_rgb_mutex.unlock();
	};
	// Do not call directly even in child
	void DepthCallback(void* _depth, uint32_t timestamp) {

		m_depth_mutex.lock();
		uint16_t* depth = static_cast<uint16_t*>(_depth);

		depthMat.data = (uchar*) depth;
		m_new_depth_frame = true;
		m_depth_mutex.unlock();

	}

	bool getVideo(Mat& output) {
		m_rgb_mutex.lock();
		if(m_new_rgb_frame) {
			cv::cvtColor(rgbMat, output, CV_RGB2BGR);
			m_new_rgb_frame = false;
			m_rgb_mutex.unlock();
			return true;
		} else {
			m_rgb_mutex.unlock();
			return false;
		}
	}

	bool getDepth(Mat& output) {
			m_depth_mutex.lock();
			if(m_new_depth_frame) {
				depthMat.copyTo(output);
				m_new_depth_frame = false;
				m_depth_mutex.unlock();
				return true;
			} else {
				m_depth_mutex.unlock();
				return false;
			}
		}

  private:
	std::vector<uint8_t> m_buffer_depth;
	std::vector<uint8_t> m_buffer_rgb;
	std::vector<uint16_t> m_gamma;
	Mat depthMat;
	Mat rgbMat;
	Mat ownMat;
	Mutex m_rgb_mutex;
	Mutex m_depth_mutex;
	bool m_new_rgb_frame;
	bool m_new_depth_frame;
};

void convertToColor(Mat_<float> D1_data, Mat_<Vec3b> D1_data_color){

	for (int j = 0; j < D1_data.cols; j++) {
		for (int i = 0; i < D1_data.rows; i++) {
			Vec3b v;

			float val = min(D1_data.at<float>(i,j) * 0.01f, 1.0f);
			if (val <= 0) v[0] = v[1] = v[2] = 0;
			else {
				float h2 = 6.0f * (1.0f - val);
				unsigned char x  = (unsigned char)((1.0f - fabs(fmod(h2, 2.0f) - 1.0f))*255);
				if (0 <= h2&&h2<1) { v[0] = 255; v[1] = x; v[2] = 0; }
				else if (1 <= h2 && h2 < 2)  { v[0] = x; v[1] = 255; v[2] = 0; }
				else if (2 <= h2 && h2 < 3)  { v[0] = 0; v[1] = 255; v[2] = x; }
				else if (3 <= h2 && h2 < 4)  { v[0] = 0; v[1] = x; v[2] = 255; }
				else if (4 <= h2 && h2 < 5)  { v[0] = x; v[1] = 0; v[2] = 255; }
				else if (5 <= h2 && h2 <= 6) { v[0] = 255; v[1] = 0; v[2] = x; }
            }
            D1_data_color.at<cv::Vec3b>(i, j) = v;
        }
    }
}

void deleteBackground(Mat_<float>* depth, Mat* result, Point* center, float thresh) {
    int i, j;
    Vec3b v, black, white;
    v[0] = 0;
    v[1] = 0;
    v[2] = 0;

    black[0] = 0;
    black[1] = 0;
    black[2] = 0;
    white[0] = 255;
    white[1] = 255;
    white[2] = 255;
    int x = 0;
    int y = 0;
    int cant = 0;
    for(i = 0; i<depth->rows; i++) {
        for(j = 0; j< depth->cols; j++) {
            if(depth->at<float>(i,j) > thresh){
                result->at<Vec3b>(i,j) = black;
            } else {
                result->at<Vec3b>(i,j) = white;
                x = x + i;
                y = y + j;
                cant++;
            }
        }

    }

    if(cant > 0) {
        *center = Point(x/cant, y/cant);
    }
}

void deleteBackground(Mat_<float>* depth, Mat* image, Mat* result, Point* center, int thresh) {
    int i, j;
    Vec3b v, black, white;
    v[0] = 0;
    v[1] = 0;
    v[2] = 0;

    black[0] = 0;
    black[1] = 0;
    black[2] = 0;
    white[0] = 255;
    white[1] = 255;
    white[2] = 255;
    int x = 0;
    int y = 0;
    int cant = 0;
    for(i = 0; i<image->rows; i++) {
        for(j = 0; j< image->cols; j++) {
            float val = min(depth->at<float>(i,j) * 0.01f, 1.0f);
            float h2 = 0;
            if(val>0)
            {
                h2 = 6.0f * (1.0f - val);
            }
            if(h2 < thresh){
                 result->at<Vec3b>(i,j) = black;
            } else {
                result->at<Vec3b>(i,j) = white;
                x = x + i;
                y = y + j;
                cant++;
            }
        }

    }

    if(cant > 0) {
        *center = Point(x/cant, y/cant);
    }
}

float distancePoints(Point p1, Point p2) {
    float x = (float)(p1.x - p2.x);
    float y = (float)(p1.y - p2.y);

    return sqrt((x * x) + (y * y));
}

Point nextPointFor(Point point, Mat* img) {

    int i,j;
    i = point.x - 1;
    while(i > 0 && img->at<uchar>(i,point.y) == 0) {
        i--;
    }
    int limitXInf = max(i,0);

    i = point.x + 1;
    while(i < img->rows && img->at<uchar>(i,point.y) == 0) {
        i++;
    }
    int limitXSup = min(i,img->rows);

    i = point.y - 1;
    while(i > 0 && img->at<uchar>(point.x,i) == 0) {
        i--;
    }
    int limitYInf = max(i,0);

    i = point.y + 1;
    while(i < img->cols && img->at<uchar>(point.x,i) == 0) {
        i++;
    }
    int limitYSup = min(i,img->cols);

    cout << "Para el point (" << point.x << "," << point.y << ") los limites son: " << limitXInf << ", " << limitXSup << ", " << limitYInf << ", " << limitYSup << endl;

    //Pongo el primer punto
    Point p = Point(limitXInf,limitYInf);
    float minDistance = distancePoints(p, point);

    for(i=limitXInf;i<limitXSup;i++) {
		for(j=limitYInf;j<limitYSup;j++) {
			if(img->at<uchar>(i,j) > 0 && distancePoints(p, Point(i,j)) < minDistance) {
			    minDistance = distancePoints(p, Point(i,j));
			    p = Point(i,j);
            }
		}
	}

    img->at<uchar>(i,j) = 0;
    return p;
}

Point nextPointFor(Point point, Mat img, int size) {

    int i,j;

    Point p = Point(-1,-1);
    float minDistance = size * size;
    float actualDistance;

    for(i=max(0,point.x-size);i<min(point.x,img.rows)+size;i++) {
		for(j=max(0,point.y-size);j<min(point.y+size,img.cols);j++) {
			if(img.at<uchar>(i,j) > 0){
                actualDistance = distancePoints(point, Point(i,j));
                if(actualDistance > 0 && actualDistance < minDistance) {
                    minDistance = actualDistance;
                    p = Point(i,j);
                }
            }
		}
	}

/*
    if(p.x = -1 && p.y == -1 && size < 30) {
        return nextPointFor(point, img, size+1);
    }
*/
    img.at<uchar>(p.x,p.y) = 0;
    return p;
}

uchar firstDerivedX(int index, Point* points, int cant) {
    if(index == cant - 1) {
        return 0;
    } else {
        return points[index+1].x - points[index].x;
    }
}

uchar secondDerivedX(int index, Point* points, int cant) {
    if(index == cant - 1 || index == 0) {
        return 0;
    } else {
        return points[index+1].x - points[index].x + points[index-1].x;
    }
}

uchar firstDerivedY(int index, Point* points, int cant) {
    if(index == cant - 1) {
        return 0;
    } else {
        return points[index+1].y - points[index].y;
    }
}

uchar secondDerivedY(int index, Point* points, int cant) {
    if(index == cant - 1 || index == 0) {
        return 0;
    } else {
        return points[index+1].y - points[index].y + points[index-1].y;
    }
}

void calculateCurvatures(Point* points, float* curvatures, int cant) {
    // k = (x'*y''-y'*x'')/((x'^2 + y'^2)^3/2);
    uchar xp, xpp, yp, ypp;
    Point p;
    for(int i=0; i<cant; i++) {
        xp = firstDerivedX(i, points, cant);
        xpp = secondDerivedX(i, points, cant);
        yp = firstDerivedY(i, points, cant);
        ypp = secondDerivedY(i, points, cant);

        curvatures[i] = ((xp * ypp) - (yp * xpp)) / (pow((pow(xp,2) + pow(yp,2)),3/2));
    }
}

bool containsFinger(list<Point>* fingers, Point point, int radius) {
    // Se obtiene un iterador al inicio de la lista
    list<Point>::iterator it = fingers->begin();

    // Buscamos un elemento dentro del radio elemento "Pedro"
    while (it != fingers->end()) {
        if(distancePoints(*it,point)<=radius)
            return true;
        it++;
    }

    return false;
}

void detectFingersByCurvature(float* curvatures, Point* points, Mat* img, int cant, list<Point>* fingers){
    int radius = 10;
    if(cant > 2) {

        for(int i = 2 ; i < cant-2 ; i++) {
            //if(curvatures[i] < curvatures[i-1] && curvatures[i] < curvatures[i+1] && curvatures[i] <= -THRESH_CURVATURE && curvatures[i] < curvatures[i+2] && curvatures[i] < curvatures[i-2]) {

            if(i>=2 && i < cant-2){
                if(curvatures[i] > curvatures[i-1] && curvatures[i] > curvatures[i+1] && curvatures[i] >= THRESH_CURVATURE && curvatures[i] > curvatures[i+2] && curvatures[i] > curvatures[i-2]) {

                    //Verifico la zona
                    if(!containsFinger(fingers, Point(points[i].y,points[i].x), radius)){
                        cout << "la curvatura es: " << curvatures[i] << endl;
                        //circle(*img, Point(points[i].y,points[i].x), radius, Scalar(200));
                        fingers->push_back(Point(points[i].y,points[i].x));
                    }
                }

            }

            img->at<uchar>(points[i].x,points[i].y) = 255;

        }
    }


}

void filterPoints(Point* points, Point* filteredPoints, int size, int mask){
    int i, j, x, y;

    //for(i=mask ; i< size-mask ; i++){
    for(i=0 ; i< size ; i++){
        x = 0;
        y = 0;
        int cant = 0;
        for(j = -mask ; j <= mask ; j++){
            if(points[(i+j+size)%size].x != 0 || points[(i+j+size)%size].y != 0) {
                cant ++;
                x = x + points[(i+j+size)%size].x;
                y = y + points[(i+j+size)%size].y;
            }
        }
        filteredPoints[i] = Point(x/cant,y/cant);
    }
}

void detectFingers(IplImage* data, Mat* result, list<Point>* fingers){
	Mat aux(data);
	Mat img = aux.clone();
	Mat img2 = aux.clone();
	int i, j, cant = 0;
    Point p;

	//Cuento la cantidad de puntos no nulos
	for(i=0;i<img.rows;i++) {
		for(j=0;j<img.cols;j++) {
			if(img.at<uchar>(i,j) > 0) {
			    if(cant == 0){
                    p = Point(i,j);
			    }
				cant++;
			}
		}
	}

    cout << "los blancos son: " << cant << endl;

	Point points [cant];

	int index = 0;

    if(cant > 0) {
        points[index] = p;
        img2.at<uchar>(p.x,p.y) = 0;
        index++;
    }
    cout << "El primer punto es (" << points[0].x << "," << points[0].y  << ")" << endl;


	//Busco el resto
	int cantPoints = 1;
	while(index<cant){
		points[index] = nextPointFor(points[index-1],img2, 4);

        //cout << "El punto " << index << " es: (" << points[index].x << "," << points[index].y  << ")" << endl;

		if(points[index].x > 0 || points[index].y > 0)
            cantPoints++;

        index++;
	}

	float curvatures[cantPoints];


	Point filteredPoints[cantPoints];

	filterPoints(points, filteredPoints, cantPoints, THRESH_MASK);

	calculateCurvatures(filteredPoints, curvatures, cantPoints);

    detectFingersByCurvature(curvatures, filteredPoints, result, cantPoints, fingers);

}

void drawResults(Mat * img, Point center, list<Point>* fingers){
    //Draw center
    circle(*img, Point(center.y,center.x), 10, Scalar(255));

    //Draw fingers
    list<Point>::iterator it = fingers->begin();

    // Buscamos un elemento dentro del radio elemento "Pedro"
    while (it != fingers->end()) {
        circle(*img, *it, 5, Scalar(200));
        it++;
    }
}

float calculateThreshDepth(Mat_<float>* depth) {
    int i, j, cant = 0;
    float threshDepth = depth->at<float>(0,0);

    float max = 0;
    float min = depth->at<float>(0,0);
    for(i = 0; i<depth->rows; i++) {
        for(j = 0; j< depth->cols; j++) {
            if(depth->at<float>(i,j) < min) {
                min = depth->at<float>(i,j);
            }

            if(depth->at<float>(i,j) > max) {
                max = depth->at<float>(i,j);
            }
        }
    }

    //Scale
    threshDepth = min + (max - min) * THRESH_PORCENT;

    return threshDepth;
}

int main(int argc, char **argv) {
	bool die(false);
	string filename("snapshot");
	string suffix(".png");
	int i_snap(0),iter(0);


	Mat_<float> depthMat(Size(640,480),CV_16UC1);
	Mat depthf  (Size(640,480),CV_8UC1);
	Mat rgbMat(Size(640,480),CV_8UC3,Scalar(0));
	Mat rgbMatWithOutBackground(Size(640,480),CV_8UC3,Scalar(0));
	Mat ownMat(Size(640,480),CV_8UC3,Scalar(0));
	Mat_<Vec3b> depthMatColor(Size(640,480));


    Freenect::Freenect freenect;
    MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);

	namedWindow("rgb",CV_WINDOW_AUTOSIZE);
	namedWindow("depth",CV_WINDOW_AUTOSIZE);
	device.startVideo();
	device.startDepth();
    Mat_<int> bn(Size(640,480));
    while (!die) {

        try {

        //Leo datos de Kinect
        device.getVideo(rgbMat);
    	device.getDepth(depthMat);

        //Scale
        depthMat = depthMat/10;

        float threshDepth = calculateThreshDepth(&depthMat);

        cout << "threshDepth: " << threshDepth << endl;
        cout << "THRESH_PORCENT: " << THRESH_PORCENT << endl;



        Point center;
        //Elimino el fondo
        //deleteBackground(&depthMat,&rgbMat,&rgbMatWithOutBackground, &center, THRESH);
        deleteBackground(&depthMat,&rgbMatWithOutBackground, &center, threshDepth);
        cout << "borre fondo" << endl;

        //Convierto a blanco y negro la imagen
        cvtColor(rgbMatWithOutBackground,bn,CV_RGB2GRAY);
        cout << "convierto profundidad a color" << endl;


        //Creo el IplImage a partir de la imagen
        IplImage *img = new IplImage(bn);

        //Creo la imagen de salida de canny
        IplImage* canny = cvCreateImage( cvSize(img->width,img->height), img->depth, img->nChannels );

        //Aplico canny
        cvCanny( img, canny, 100, 200, 3 );

        //Mat fingers(canny);
        Mat result  (Size(640,480),CV_8UC1);

        list<Point> fingers;

        //Calculo curvatura
        detectFingers(canny,&result, &fingers);

        drawResults(&result, center, &fingers);



        cv::imshow("blanco y negro", bn);
        cv::imshow("rgb", rgbMat);
        cvShowImage("canny", canny);
        cv::imshow("fingers", result);
        cout << "muestro todo" << endl;
    	//depthMat.convertTo(depthf, CV_8UC1, 255.0/2048.0 );



        convertToColor(depthMat,depthMatColor);

        cv::imshow("depth",depthMatColor);

        }
        catch( char * str )
        {
            cout << "Exception raised: " << str << '\n';
        }

        //bn.release();

		char k = cvWaitKey(5);
		if( k == 27 ){
		    cvDestroyWindow("rgb");
		    cvDestroyWindow("depth");
			break;
		}
		if( k == 8 ) {
			std::ostringstream file;
			file << filename << i_snap << suffix;
			cv::imwrite(file.str(),rgbMat);
			i_snap++;
		}

    }


   	device.stopVideo();
	device.stopDepth();

	return 0;
}
