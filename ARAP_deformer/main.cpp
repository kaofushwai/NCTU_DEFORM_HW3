#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <set>
#include <numeric>
#include <iostream>
#include <limits>
#include <Windows.h>
#include <gl/GL.h>
#include <glut.h>

#include "glm.h"
#include "mtxlib.h"
#include "trackball.h"
#include "LeastSquaresSparseSolver.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;

// ----------------------------------------------------------------------------------------------------
// global variables

_GLMmodel *mesh;

int WindWidth, WindHeight;
int last_x , last_y;
int select_x, select_y;

typedef enum { SELECT_MODE, DEFORM_MODE } ControlMode;
ControlMode current_mode = SELECT_MODE;

vector<float*> colors;
vector<vector<int> > handles;

std::set<GLuint> *neighbors;

Eigen::Matrix<GLfloat, 3, Eigen::Dynamic> *eij, *epij; // eij, e'ij
Eigen::Matrix3f *Ri;
GLfloat *original_vertices;
GLfloat **w;
GLuint numvertices, total_rows;
float **b;
LeastSquaresSparseSolver solver;
int selected_handle_id = -1;
bool deform_mesh_flag = false;

// ----------------------------------------------------------------------------------------------------
// render related functions

void Reshape(int width, int height)
{
	int base = min(width , height);

	tbReshape(width, height);
	glViewport(0 , 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0,(GLdouble)width / (GLdouble)height , 1.0, 128.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, -3.5);

	WindWidth = width;
	WindHeight = height;
}

void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPushMatrix();
	tbMatrix();

	// render solid model
	glEnable(GL_LIGHTING);
	glColor3f(1.0 , 1.0 , 1.0f);
	glPolygonMode(GL_FRONT_AND_BACK , GL_FILL);
	glmDraw(mesh , GLM_SMOOTH);

	// render wire model
	glPolygonOffset(1.0 , 1.0);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glLineWidth(1.0f);
	glColor3f(0.6 , 0.0 , 0.8);
	glPolygonMode(GL_FRONT_AND_BACK , GL_LINE);
	glmDraw(mesh , GLM_SMOOTH);

	// render handle points
	glPointSize(10.0);
	glEnable(GL_POINT_SMOOTH);
	glDisable(GL_LIGHTING);
	glBegin(GL_POINTS);
	for(int handleIter=0; handleIter<handles.size(); handleIter++)
	{
		glColor3fv(colors[ handleIter%colors.size() ]);
		for(int vertIter=0; vertIter<handles[handleIter].size(); vertIter++)
		{
			int idx = handles[handleIter][vertIter];
			glVertex3fv((float *)&mesh->vertices[3 * idx]);
		}
	}
	glEnd();

	glPopMatrix();

	glFlush();  
	glutSwapBuffers();
}

// ----------------------------------------------------------------------------------------------------
// mouse related functions

vector3 Unprojection(vector2 _2Dpos)
{
	float Depth;
	int viewport[4];
	double ModelViewMatrix[16];    // Model_view matrix
	double ProjectionMatrix[16];   // Projection matrix

	glPushMatrix();
	tbMatrix();

	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_MODELVIEW_MATRIX, ModelViewMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, ProjectionMatrix);

	glPopMatrix();

	glReadPixels((int)_2Dpos.x , viewport[3] - (int)_2Dpos.y , 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &Depth);

	double X = _2Dpos.x;
	double Y = _2Dpos.y;
	double wpos[3] = {0.0 , 0.0 , 0.0};

	gluUnProject(X , ((double)viewport[3] - Y) , (double)Depth , ModelViewMatrix , ProjectionMatrix , viewport, &wpos[0] , &wpos[1] , &wpos[2]);

	return vector3(wpos[0] , wpos[1] , wpos[2]);
}

vector2 projection_helper(vector3 _3Dpos)
{
	int viewport[4];
	double ModelViewMatrix[16];    // Model_view matrix
	double ProjectionMatrix[16];   // Projection matrix

	glPushMatrix();
	tbMatrix();

	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_MODELVIEW_MATRIX, ModelViewMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, ProjectionMatrix);

	glPopMatrix();

	double wpos[3] = {0.0 , 0.0 , 0.0};
	gluProject(_3Dpos.x, _3Dpos.y, _3Dpos.z, ModelViewMatrix, ProjectionMatrix, viewport, &wpos[0] , &wpos[1] , &wpos[2]);

	return vector2(wpos[0], (double)viewport[3]-wpos[1]);
}

void mouse(int button, int state, int x, int y)
{
	tbMouse(button, state, x, y);

	if( current_mode==SELECT_MODE && button==GLUT_RIGHT_BUTTON )
	{
		if(state==GLUT_DOWN)
		{
			select_x = x;
			select_y = y;
		}
		else
		{
			vector<int> this_handle;

			// project all mesh vertices to current viewport
			for(int vertIter=0; vertIter<mesh->numvertices; vertIter++)
			{
				vector3 pt(mesh->vertices[3 * vertIter + 0] , mesh->vertices[3 * vertIter + 1] , mesh->vertices[3 * vertIter + 2]);
				vector2 pos = projection_helper(pt);

				// if the projection is inside the box specified by mouse click&drag, add it to current handle
				if(pos.x>=select_x && pos.y>=select_y && pos.x<=x && pos.y<=y)
				{
					this_handle.push_back(vertIter);
				}
			}
			handles.push_back(this_handle);
		}
	}
	// select handle
	else if( current_mode==DEFORM_MODE && button==GLUT_RIGHT_BUTTON && state==GLUT_DOWN )
	{
		// project all handle vertices to current viewport
		// see which is closest to selection point
		double min_dist = 999999;
		int handle_id = -1;
		for(int handleIter=0; handleIter<handles.size(); handleIter++)
		{
			for(int vertIter=0; vertIter<handles[handleIter].size(); vertIter++)
			{
				int idx = handles[handleIter][vertIter];
				vector3 pt(mesh->vertices[3 * idx + 0] , mesh->vertices[3 * idx + 1] , mesh->vertices[3 * idx + 2]);
				vector2 pos = projection_helper(pt);

				double this_dist = sqrt((double)(pos.x-x)*(pos.x-x) + (double)(pos.y-y)*(pos.y-y));
				if(this_dist<min_dist)
				{
					min_dist = this_dist;
					handle_id = handleIter;
				}
			}
		}

		selected_handle_id = handle_id;
		deform_mesh_flag = true;
	}

	if(button == GLUT_RIGHT_BUTTON && state == GLUT_UP)
		deform_mesh_flag = false;

	last_x = x;
	last_y = y;
}

void motion(int x, int y)
{
	tbMotion(x, y);

	// if in deform mode and a handle is selected, deform the mesh
	if( current_mode==DEFORM_MODE && deform_mesh_flag==true )
	{
		matrix44 m;
		vector4 vec = vector4((float)(x - last_x) / 1000.0f , (float)(y - last_y) / 1000.0f , 0.0 , 1.0);

		gettbMatrix((float *)&m);
		vec = m * vec;

		// deform handle points
		for(int vertIter=0; vertIter<handles[selected_handle_id].size(); vertIter++)
		{
			int idx = handles[selected_handle_id][vertIter];
			vector3 pt(mesh->vertices[3*idx+0]+vec.x, mesh->vertices[3*idx+1]+vec.y, mesh->vertices[3*idx+2]+vec.z);
			mesh->vertices[3 * idx + 0] = pt[0];
			mesh->vertices[3 * idx + 1] = pt[1];
			mesh->vertices[3 * idx + 2] = pt[2];
		}
	}

	last_x = x;
	last_y = y;
}

// helper functions

void print_mode() {
	switch (current_mode)
	{
	case DEFORM_MODE:
		puts("current mode=DEFORM_MODE");
		break;
	case SELECT_MODE:
		puts("current mode=SELECT_MODE");
		break;
	}
}

// ----------------------------------------------------------------------------------------------------
// keyboard related functions
void compute_Ri();
void compute_p_prime();
void keyboard(unsigned char key, int x, int y )
{
	GLuint numcontrol = 0;
	GLuint idx = numvertices + 1;
	switch(key)
	{
	case 'd':
		current_mode = DEFORM_MODE;
		// create solver
		
		for (auto handle : handles) {
			numcontrol += handle.size();
		}
		total_rows = numvertices + numcontrol + 1;
		solver.Create(total_rows, numvertices + 1, 3);

		solver.AddSysElement(0, 0, 1.0f);
		for (int i = 1; i <= numvertices; ++i) {
			solver.AddSysElement(i, i, neighbors[i].size()); // assume wij = 1
			for (auto p : neighbors[i]) {
				solver.AddSysElement(i, p, -1.0f);
			}
		}

		idx = numvertices + 1;
		for (auto handle : handles) {
			for (auto p : handle) {
				solver.AddSysElement(idx++, p, 1.0f);
			}
		}

		b = new float*[3];
		for (int i = 0; i < 3; ++i) {
			b[i] = new float[total_rows];
		}
		solver.SetRightHandSideMatrix(b);

		solver.CholoskyFactorization();

		
		break;
	
	case 's':
		current_mode = SELECT_MODE;
		solver.ResetSolver(0, 0, 0);
		for (int i = 0; i < 3; ++i) {
			delete[] b[i];
		}
		delete[] b;
		break;

	case 'c':
		for (int i = 0; i < 1; ++i) {
			compute_Ri();
			compute_p_prime();
		}
		break;
	default:
		break;
	}
	print_mode();
}

// ----------------------------------------------------------------------------------------------------
// main function

void timf(int value)
{
	glutPostRedisplay();
	glutTimerFunc(1, timf, 0);
}

void compute_edge(Eigen::Matrix<GLfloat, 3, Eigen::Dynamic> *ei_j, GLfloat *vertices, GLuint numvertices) {
	for (int i = 1; i <= numvertices; ++i) {
		int idx = 0;
		ei_j[i].resize(3, neighbors[i].size());
		for (auto j : neighbors[i]) {
			Eigen::Vector3f pi = Eigen::Vector3f(vertices[i * 3 + 0], vertices[i * 3 + 1], vertices[i * 3 + 2]);
			Eigen::Vector3f pj = Eigen::Vector3f(vertices[j * 3 + 0], vertices[j * 3 + 1], vertices[j * 3 + 2]);
			ei_j[i].col(idx++) = (pi - pj);
		}
	}
}

void compute_Ri() {
	compute_edge(epij, mesh->vertices, numvertices);
	for (int i = 1; i <= numvertices; ++i) {
		Eigen::Matrix3f Si = eij[i] * epij[i].transpose(); // assume wij = 1
		Eigen::JacobiSVD<Eigen::Matrix3f> svd(Si, Eigen::ComputeFullU | Eigen::ComputeFullV);
		// note that svd.matrixV() is actually V^T!!
		Ri[i] = (svd.matrixV() * svd.matrixU().transpose()); // Ri
		if (Ri[i].determinant() < 0) {
			int min_idx = 0;
			for (int i = 1; i < 3; ++i) {
				if (svd.singularValues()[i] < svd.singularValues()[min_idx]) {
					min_idx = i;
				}
			}
			Eigen::Matrix3f u = svd.matrixU();
			u.col(min_idx) = -u.col(min_idx);
			Ri[i] = (svd.matrixV() * u.transpose());
		}
	}
}

void compute_p_prime() {
	b[0][0] = b[1][0] = b[2][0] = 0;
	for (int i = 1; i <= numvertices; ++i) {
		int idx = 0;
		Eigen::Vector3f bv(0, 0, 0);
		for (auto p : neighbors[i]) {
			bv += 0.5 * (Ri[i] + Ri[p]) * eij[i].col(idx++);
		}
		for (int j = 0; j < 3; ++j) {
			b[j][i] = bv[j];
		}
	}

	GLuint idx = numvertices + 1;
	for (auto handle : handles) {
		for (auto p : handle) {
			for (int i = 0; i < 3; ++i) {
				b[i][idx] = mesh->vertices[3 * p + i];
			}
			idx++;
		}
	}
	std::cout << total_rows << std::endl;
	std::cout << idx << std::endl;
	solver.SetRightHandSideMatrix(b);
	solver.CholoskySolve(0);
	solver.CholoskySolve(1);
	solver.CholoskySolve(2);

	for (int i = 1; i <= numvertices; ++i) {
		for (int j = 0; j<3; ++j) {
			mesh->vertices[3 * i + j] = solver.GetSolution(j, i);
		}
	}
}


int main(int argc, char *argv[])
{
	// compute SVD decomposition of a matrix m
	// SVD: m = U * S * V^T
	/*Eigen::MatrixXf m = Eigen::MatrixXf::Random(3,2);
	cout << "Here is the matrix m:" << endl << m << endl;
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
	const Eigen::Matrix3f U = svd.matrixU();
	// note that this is actually V^T!!
	const Eigen::Matrix3f V = svd.matrixV();
	const Eigen::VectorXf S = svd.singularValues();*/
	std::cout << "initializing" << std::endl;

	WindWidth = 800;
	WindHeight = 800;

	GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 1.0 };
	GLfloat light_diffuse[] = { 0.8, 0.8, 0.8, 1.0 };
	GLfloat light_specular[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat light_position[] = { 0.0, 0.0, 1.0, 0.0 };

	// color list for rendering handles
	float red[] = { 1.0, 0.0, 0.0 };
	colors.push_back(red);
	float yellow[] = { 1.0, 1.0, 0.0 };
	colors.push_back(yellow);
	float blue[] = { 0.0, 1.0, 1.0 };
	colors.push_back(blue);
	float green[] = { 0.0, 1.0, 0.0 };
	colors.push_back(green);

	glutInit(&argc, argv);
	glutInitWindowSize(WindWidth, WindHeight);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutCreateWindow("ARAP");

	glutReshapeFunc(Reshape);
	glutDisplayFunc(Display);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(keyboard);
	glClearColor(0, 0, 0, 0);

	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	glEnable(GL_LIGHT0);
	glDepthFunc(GL_LESS);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glEnable(GL_COLOR_MATERIAL);
	tbInit(GLUT_LEFT_BUTTON);
	tbAnimate(GL_TRUE);

	glutTimerFunc(40, timf, 0); // Set up timer for 40ms, about 25 fps

	// load 3D model
	mesh = glmReadOBJ("../data/Armadillo.obj");

	glmUnitize(mesh);
	glmFacetNormals(mesh);
	glmVertexNormals(mesh, 90.0);

	// pre compute
	// save a mesh data copy
	numvertices = mesh->numvertices;
	original_vertices = new GLfloat[3 * (mesh->numvertices + 1)];
	for (int i = 0; i <= 3 * numvertices; ++i) {
		original_vertices[i] = mesh->vertices[i];
	}
	eij = new Eigen::Matrix<GLfloat, 3, Eigen::Dynamic>[numvertices + 1];
	epij = new Eigen::Matrix<GLfloat, 3, Eigen::Dynamic>[numvertices + 1];

	Ri = new Eigen::Matrix3f[numvertices + 1];

	// neighbor
	{
		neighbors = new std::set<GLuint>[numvertices + 1];
		for (int t = 0; t < mesh->numtriangles; ++t) {
			for (int i = 0; i < 3; ++i) {
				// p_i: index of pi
				GLuint p_i = mesh->triangles[t].vindices[i];
				for (int j = 0; j < 3; ++j) {
					GLuint p_j = mesh->triangles[t].vindices[j];
					if (i != j) {
						neighbors[p_i].insert(p_j);
					}
				}
			}
		}
	}

	// eij
	if (1) {
		compute_edge(eij, original_vertices, numvertices);
	}

	print_mode();
	glutMainLoop();
	
	return 0;

}