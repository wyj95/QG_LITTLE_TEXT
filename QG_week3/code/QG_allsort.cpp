#include<sort.h>
/**
 *  @name        : void InsertSort(int *a,int n);
 *  @description : ���������㷨
 *  @param       : ����ָ�� a, ���鳤�� n
 */
void InsertSort(int *a, long int n)
{
	for(long int i=0; i<n-1; i++)
	{
		int max=a[i],cj=i;/* cj��¼���ֵ��Ӧ��λ�� */ 
		for(int j=i; j<n; j++)
			if(a[j] > max)
			{
				cj = j;
				max = a[j];
			}
		int t = a[cj];/* ��¼�õ� */
		/* ����ʵ�ֲ������ */
		for(long int j=cj; j>i; j--)
			a[j] = a[j-1];
		a[i] = t;
	}
}

/**
 *  @name        : void MergeArray(int *a,int begin,int end);
 *  @description : �鲢���򣨺ϲ����飩
 *  @param       : ����ָ��a���������begin�������յ�end
 */
void MergeArray(int *a,long int begin,long int end)
{
	if(begin >= end)
		return;
	long int mid=end + begin;
	mid /= 2;                                           /* �е� */
	int *p=(int *)malloc(sizeof(int)*(end - begin + 1));/* ������� */
	long int b=begin, e=mid+1;							/* b��begin��������mid��e��mid��������end */ 
	long int i=0;                                       /* ���ڸ���p��¼���� */
	while(b<=mid && e<=end)
	{
		if(a[b] < a[e])
			*(p + i++)=a[ e++];
		else
			*(p + i++)=a[ b++];
	}
	/* ����Ƚ�������p */ 
	while(b <= mid)
		*(p + i++)=a[ b++];
	while(e <= end)
		*(p + i++)=a[ e++];
	/* ������µ�Ԫ�� */ 
	for(long int j=0; j<i; )
		a[begin + j] = *(p + j++);
	free(p);
	/* ��������Ԫ���ó������ͷ�p */ 
}


/**
 *  @name        : void MergeSort(int *a,int begin,int end);
 *  @description : �鲢����
 *  @param       : ����ָ��a���������begin�������յ�end
 */
void MergeSort(int *a,long int begin,long int end)
{
	if(begin >= end)
		return;
	long int mid=end + begin;
	mid /= 2;
	MergeSort(a, begin, mid);
	MergeSort(a, mid+1, end);
	MergeArray(a, begin, end);
	/* �ȹ�� */ 
}


/**
 *  @name        : void QuickSortRecursion(int *a, int begin, int end);
 *  @description : �������򣨵ݹ�棩
 *  @param       : ����ָ��a���������begin�������յ�end
 */
void QuickSortRecursion(int *a, long int begin, long int end)
{
	if(begin >= end)
		return;
	long int i=begin, j=end;
	/* i���ڴӿ�ͷ������j���ڴ�ĩβ��ʼ���� */
	int t=a[i];/* �洢������Ԫ�� */ 
	while(i < j)
	{
		while(a[j]<t && j>i)
			j--;
		if(j > i)
			a[i++]=a[j];
		while(a[i]>t && j>i)
			i++;
		if(j > i)
			a[j--]=a[i];
	}
	a[i] = t;
	QuickSortRecursion(a, begin, i-1);
	QuickSortRecursion(a, i+1, end);
	/* ��ʱiӦ�õ���j����Ҳ����j */ 
}


/**
 *  @name        : void QuickSort(int *a,long int begin, long int end)
 *  @description : �������򣨷ǵݹ�棩
 *  @param       : ����ָ��a�����鳤��size
 */
void QuickSort(int *a,long int begin, long int end)
{
	long int partition;
	if(begin >= end)
		return;
	while(begin < end)
	{
		partition = Partition(a, begin, end);
		QuickSort(a, begin, partition-1);
		begin = partition+1;
	}
}


/**
 *  @name        : long int Partition(int *a, int begin, int end)
 *  @description : �������������ţ�
 *  @param       : ����ָ��a���������begin�������յ�end
 */
long int Partition(int *a, long int begin, long int end)
{
	long int i=begin, j=end;	/* i���ڴӿ�ͷ������j���ڴ�ĩβ��ʼ���� */
	int t=a[i];					/* �洢������Ԫ�� */ 
	while(i < j)
	{
		while(a[j]<t && j>i)
			j--;
		if(j > i)
			a[i++]=a[j];
		while(a[i]>t && j>i)
			i++;
		if(j > i)
			a[j--]=a[i];
	}
	a[i] = t;
	return i;
	/* �������� */ 
}


/**
 *  @name        : void CountSort(int *a, int size)
 *  @description : ��������
 *  @param       : ����ָ��a�����鳤��size
 */
void CountSort(int *a, long int size)
{
	int max=a[0], min=a[0];
	long int i,j;
	for(i=1; i<size; i++)
		if(a[i] < min)
			min = a[i];
		else if(a[i] > max)
			max = a[i];
	int length=max - min + 1;
	/* ���max��min��ȷ���������鳤�� */ 
	int *p=(int *)calloc(length, sizeof(int));
	/* ���ڴ洢������Ĭ����0 */ 
	for(i=0; i<size; i++)
		*(p + a[i] - min) += 1;
	j = 0;
	for(i=length-1; i>=0; i--)
		while(*(p+i) != 0)
		{
			*(p+i) -= 1;
			a[j++] = i+min;
		}
	/* �Ӽ��������ж������� */ 
	free(p);
}
 
/**
 *  @name        : void RadixCountSort(int *a,int size)
 *  @description : ������������
 *  @param       : ����ָ��a�����鳤��size
 */
void RadixCountSort(int *a,long int size)
{
	long int count1=10,count2=1;
	/* ���ڼ����Ӧλ�� �� 98%count1 / count2 == 8 �ó�count2��Ӧ��λ�� */
	int max, min, temp;
	int *p[10];
	/* 10��Ͱ */ 
	for(int i=0; i<10; i++)
		p[i] = (int*)calloc(size, sizeof(int));
	long int number[10] = {0};
	/* p[10]���ڼ�������Ͱ�������� */ 
	while(count1 <= 100000) /*�˴�ѭ������Ҳ�ɸ��������ڲ�ͬ���ݴ�С�޸Ļ��������ѭ������ʱ�ж� */ 
	{
		for(long int i=0; i<size; i++)
		{
			temp = (int)(a[i]%count1 / count2);
			/* �ó����������Ǹ�Ͱ��temp�� */ 
			p[temp][ number[temp]++ ] = a[i];
			/* ������� */
		}
		temp = 0;
		long int k = 0;/* ����a��Ͱ��ȡ������ */ 
		for(int i=9; i >= 0; i--)
		{
			for(long int j=0; j<number[i]; j++)
			{
				a[k++] = p[i][j];
			}
			number[i] = 0;
		}
		/* ��Ͱ */ 
		count1 *= 10;
		count2 *= 10; 
	}
	for(int i=0; i<10; i++)
		free(p[i]);
}


