#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define PI 3.14159265358979323846  /* pi */
#include <immintrin.h>

void inicializar_onda(
    float *u, 
    int Nx, 
    float L, 
    float r,
    float amp, //amplitud del pulso
    float sigma,
    float x0){
    
    float dx;
    float x_i;

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

//Acá se debe paralelizar código
void actualizar_onda( 
    float *u,
    int j,      //paso de tiempo
    int Nx,     //cantidad de puntos espaciales
    float r){
    
    //Definir constantes en registros AVX
    __m256 dosvx = _mm256_set1_ps(2.0f);
    __m256 r2vx  = _mm256_set1_ps(r*r);

    //Paralelización
    
    // Suponiendo que i avanza de 8 en 8 (step de un registro float de 256 bits)
    for (int i = 1; i < Nx - 8; i += 8){
        // --- Fila j-1  ---
        __m256 u_mid   = _mm256_loadu_ps(&u[i + Nx*(j-1)]);     
        __m256 u_right = _mm256_loadu_ps(&u[(i+1) + Nx*(j-1)]); 
        __m256 u_left  = _mm256_loadu_ps(&u[(i-1) + Nx*(j-1)]); 

        // --- Fila j-2 ---
        __m256 u_prev_mid   = _mm256_loadu_ps(&u[i + Nx*(j-2)]);     

        // --- Cálculo del término entre paréntesis: (u_right - 2*u_mid + u_left) ---
        __m256 parentesis = _mm256_sub_ps(u_right, _mm256_mul_ps(dosvx, u_mid));
        parentesis = _mm256_add_ps(parentesis, u_left);

        // --- Cálculo final ---
        // 2*u_mid - u_prev_mid + r^2 * parentesis
        __m256 term1 = _mm256_sub_ps(_mm256_mul_ps(dosvx, u_mid), u_prev_mid);
        __m256 result = _mm256_add_ps(term1, _mm256_mul_ps(r2vx, parentesis));

        // Guardar en la fila actual j
        _mm256_storeu_ps(&u[i + Nx*j], result);
    }
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
    float r = 1.0;
    float L = 1.0;
    float amp = 1.0; 
    float sigma = 0.003;
    float x0 = 0.5;
    float *u = malloc(Nx*Nt*sizeof(float));

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