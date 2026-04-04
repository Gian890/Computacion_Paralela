#include <stdio.h>
#include <math.h>
#include <stdlib.h>
# define PI 3.14159265358979323846  /* pi */


void inicializar_onda(
    double *u, 
    int Nx, 
    double L, 
    double r,
    double amp, //amplitud del pulso
    double sigma,
    double x0){
    
    double dx;
    double x_i;

    dx = L/(Nx - 1);
    
    //Inicializacion para primer instante de tiempo
    for (int i = 1; i<(Nx-1) ; i++){
        x_i = i*dx;
        u[i] = amp*exp(-pow(x_i-x0,2)/(2*sigma));
    }

    //Inicialización para segundo instante de tiempo
    for (int i = 1; i<Nx; i++)
        u[i + Nx] = u[i] - r*(u[i+1] - u[i]);  
}


void actualizar_onda(
    double *u,
    int j,     //paso de tiempo
    int Nx,     //cantidad de puntos espaciales
    double r){
    
    for (int i = 1; i < Nx-1 ;i++)
        u[i + Nx*j] = 2*u[i + Nx*(j-1)] - u[i + Nx*(j-2)] + r*r*(u[(i+1) + Nx*(j-1)] - 2*u[i + Nx*(j-1)] + u[(i-1) + Nx*(j-1)]);
} 
/*  Precauciones a considerar:
    1) Actualizar onda depende del paso anterior de tiempo.
    2) Se define la actualización de la onda en base al número de Courant r,
    en ninguna instancia se pide como entrada el tiempo total de simulación 
    ni la velocidad de la onda, estos vienen implícitos en la expresión 
    r = (c*dt)/dx.      */
int main(){
    
    //Acá definimos los parametros de nuestra simulación
    int Nx = 1000;
    int Nt = 2000;
    double r = 1.0;
    double L = 1.0;
    double amp = 1.0; 
    double sigma = 0.003;
    double x0 = 0.5;
    double *u = malloc(Nx*Nt*sizeof(double));

    inicializar_onda(u, Nx, L, r, amp, sigma, x0);
    
    //Desarrollo de simulación (Ya se tienen los valores para j=0 y j=1)
    for (int j = 2; j < Nt; j++)
        actualizar_onda(u, j, Nx, r);
       
//Guardado de datos

    FILE *archivo;
    archivo = fopen("datos.csv", "w");
    
    if (archivo == NULL) {
        printf("Error al abrir el archivo.\n");
        return 1;
    }
    for (int j = 0; j < Nt; j++){
        for (int i = 0; i < Nx-1; i++)
            fprintf(archivo, "%f,", u[i + Nx*j]);    
            fprintf(archivo, "%f\n", u[(Nx-1) + Nx*j]);
    }
    fclose(archivo);

    printf("Datos guardados exitosamente.\n");

    free(u);
    return 0;

}   
