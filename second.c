#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>

#define KILLED_PROCESS 1
MPI_Comm main_comm;

void itoa(int n, char s[])
{
    int i = 0;
    do {
        s[i++] = n % 10 + '0';
    } while ((n /= 10) > 0);
    s[i] = '\0';
    int j, k;
    char c;
    for (j = 0, k = strlen(s) - 1; j < k; j++, k--) {
        c = s[j];
        s[j] = s[k];
        s[k] = c;
    }
}


int PROCESS_RANK, PROCESS_NUM;
char filename[10];
unsigned error_occured = 0;
MPI_Comm main_comm;

static void err_handler(MPI_Comm *pcomm, int *perr, ...) {
    error_occured = 1;
    int err = *perr;
    char errstr[MPI_MAX_ERROR_STRING];
    int size, nf, len;
    MPI_Group group_f;

    MPI_Comm_size(main_comm, &size);
    MPIX_Comm_failure_ack(main_comm);
    MPIX_Comm_failure_get_acked(main_comm, &group_f);
    MPI_Group_size(group_f, &nf);
    MPI_Error_string(err, errstr, &len);
    printf("\nRank %d / %d: Notified of error %s. %d found dead\n", PROCESS_RANK, size, errstr, nf);

    // создаем новый коммуникатор без вышедшего из строя процесса
    MPIX_Comm_shrink(main_comm, &main_comm);
    MPI_Comm_rank(main_comm, &PROCESS_RANK);
    MPI_Comm_size(main_comm, &size);
    // printf ("Amount of processes in communicator without failed processes: %d\n", size);
    MPI_Barrier(main_comm);
    itoa(PROCESS_RANK, filename);
    // printf("entered %s\n", filename);
    strcat(filename, ".txt");
}

int N = 15;

#define NN 15

float Matrix[NN][NN], Vector[NN], Res[NN];
int i, j;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &PROCESS_NUM);
    MPI_Comm_rank(MPI_COMM_WORLD, &PROCESS_RANK);
    main_comm = MPI_COMM_WORLD;
    // устанавливаем обработчик ошибок
    MPI_Errhandler errh;
    MPI_Comm_create_errhandler(err_handler, &errh);
    MPI_Comm_set_errhandler(main_comm, errh);
    MPI_Barrier(main_comm);
    // для каждого процесса формируем имя файла для записи данных контрольных точек
    itoa(PROCESS_RANK, filename);
    strcat(filename, ".txt");
    
    double s, e;
    int curr_row;
    //printf("%d", 1);
    if (PROCESS_RANK == 0) {
        for(i = 0; i <= N-1; i++) {
                Vector[i] = 1.0 * i;
                Res[i] = 5.0;
                for (j = 0; j <= N - 1; j++) {
                    Matrix[i][j] = i + j + 1.0;
                }
    
        }
    }
    
    int new_row = N;
    for (int i = 0; i < PROCESS_RANK; ++i) {
        new_row -= new_row / ((PROCESS_NUM - 1) - i);
    }
    curr_row = 2; // new_row / ((PROCESS_NUM - 1) - PROCESS_RANK);
    
    float part_matr[curr_row][N], part_ans[curr_row];

    printf("\nRANK %d, CURR_ROW %d\n", PROCESS_RANK, curr_row);

    s = MPI_Wtime();
    int scatt_num[(PROCESS_NUM - 1)];
    int scatt_ind[(PROCESS_NUM - 1)];
    
    int new_rows = N;
    //printf("\n N %d\n", N);
    
    
    MPI_Bcast(Vector, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    scatt_num[0] = curr_row * N;
    scatt_ind[0] = 0;
    curr_row = N / (PROCESS_NUM - 1);
    
    for (int i = 1; i < (PROCESS_NUM - 1); ++i) {
        new_rows -= curr_row;
        curr_row = new_rows /((PROCESS_NUM - 1) - i);
        scatt_num[i] = curr_row * N;
        scatt_ind[i] = scatt_ind[i - 1] + scatt_num[i - 1];
    }


    if (PROCESS_RANK != PROCESS_NUM - 1) {
        printf("\n\nRANK: %d %d\n\n", PROCESS_RANK, PROCESS_NUM);
        
        for (int i = 0; i < (PROCESS_NUM - 1); ++i){
            printf("\nscatter ====== %d, %d\n", scatt_num[i], scatt_ind[i]);
        }
        MPI_Scatterv(Matrix, scatt_num, scatt_ind, MPI_FLOAT,
                     part_matr, scatt_num[PROCESS_RANK], MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        FILE *fp = fopen(filename, "w");

        for (int i = 0; i < curr_row; ++i) {
            for (int j = 0; j < N; ++j) {
                // printf("\n pos %d, %d\n", i, j);
                fprintf(fp, "%f ", part_matr[i][j]);
            }

        }
        fclose(fp);
    }

    if (PROCESS_RANK == KILLED_PROCESS) {
        raise(SIGKILL);
    }

    // printf("Rank %d / %d: Stayin' alive!\n", PROCESS_RANK, PROCESS_NUM - 1);


    checkpoint:
    MPI_Barrier(main_comm);

    FILE *fp = fopen(filename, "r");
    for (int i = 0; i < curr_row; ++i) {
        for (int j = 0; j< N; ++j) {
            fscanf(fp, "%f", &part_matr[i][j]);
            // printf("old element %f ", part_matr[i][j]);
        }
    }
    fclose(fp);

    printf("Current row %d\n", curr_row);

    for (i = 0; i < curr_row; ++i) {
        part_ans[i] = 0.0;
        for (j = 0; j < N; ++j) {
            part_ans[i] += part_matr[i][j] * Vector[j];
            if (error_occured) {
                
                error_occured = 0;
                goto checkpoint;
            }
        }
        printf("\n i = %d part_ans i %f\n", i, part_ans[i]);
    }

    int gath_num[PROCESS_NUM - 1];
    int gath_ind[PROCESS_NUM - 1];
    new_rows = N;
    gath_num[0] = N / (PROCESS_NUM - 1);
    gath_ind[0] = 0;
    for (int i = 1; i < (PROCESS_NUM - 1); ++i) {
        new_rows -= gath_num[i - 1];
        gath_num[i] = new_rows / ((PROCESS_NUM - 1) - i);
        gath_ind[i] = gath_ind[i - 1] + gath_num[i - 1];
    }
    for (int i = 0; i < (PROCESS_NUM - 1); ++i){
            printf("\ngather ====== %d, %d\n", gath_num[i], gath_ind[i]);
    }
    

    MPI_Gatherv(part_ans, gath_num[PROCESS_RANK], MPI_FLOAT, Res, gath_num, gath_ind, MPI_FLOAT, 0, MPI_COMM_WORLD);

    e = MPI_Wtime();
    
    if (PROCESS_RANK == 0) {
        for (int i = 0; i < (10); ++i){
            printf("result ====== %f\n", Res[i]);
        }
    
    }
    
    printf("RANK END %d\n", PROCESS_RANK);
    if (PROCESS_RANK == 0) {
        printf("N: %d\n", N);
        printf("Process number: %d\n", (PROCESS_NUM - 1));
        printf("TIME: %06.6fs\n", e - s);
        printf("\n");
    }
    MPI_Finalize();
    return 0;
}
